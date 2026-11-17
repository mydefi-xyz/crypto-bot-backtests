"""Momentum rotation backtest (cross-sectional momentum, market neutral, one account).

Every `rebalance` hours: rank the universe by return over `lookback` hours (optionally divided by
volatility), go long the top N and short the bottom N with equal notional. Gross exposure = pool x gross.
The ranking uses the close `--exec-lag` hours before the trade (default 1 = next hourly close).

The previous year is NOT out-of-sample: the setup was chosen on the same two years.

    python rotation.py                      # last 12 months + the 12 months before
    python rotation.py --rebalance 72       # slow variant, writes rotation_*_rb72.*
    python rotation.py --lookback 72 --rebalance 24 --n 3
    python rotation.py --fee 0.045 --slippage 0.03
    python rotation.py --exec-lag 0         # old behaviour: rank and trade at the same close
"""
from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data import HOUR, load_funding, load_klines, report_dir

UNIVERSE = "BTC,ETH,SOL,BNB,XRP,DOGE,ADA,AVAX,LINK,SUI,LTC,DOT,BCH,TRX,TON"


@dataclass(frozen=True)
class RotCfg:
    lookback_h: int = 168        # 7 days
    rebalance_h: int = 8
    n: int = 3                   # long top n, short bottom n
    gross: float = 1.0           # total long + short notional / equity
    vol_adj: bool = True         # rank by return / volatility
    mode: str = "momentum"       # "momentum" (long winners) | "reversal" (long losers)
    fee: float = 0.00045         # taker fee per trade (Hyperliquid base tier)
    slippage: float = 0.0003
    band: float = 0.25           # don't resize a kept position unless off by more than this
    pool: float = 2000.0
    exec_lag_h: int = 1          # hours between ranking close and execution (0 = same close)

    @property
    def label(self) -> str:
        return (f"{self.mode} LB {self.lookback_h // 24 if self.lookback_h >= 24 else self.lookback_h}"
                f"{'d' if self.lookback_h >= 24 else 'h'} / RB {self.rebalance_h}h / N {self.n}"
                f"{' / vol-adj' if self.vol_adj else ''}")


# --------------------------------------------------------------------------- data
def load_universe(coins, s_ms, e_ms, funding_src):
    closes, funds = {}, {}
    for c in coins:
        k = load_klines(c, "1h", s_ms, e_ms)
        if k.empty:
            print(f"  {c}: keine Daten, uebersprungen")
            continue
        closes[c] = pd.Series(k["close"].to_numpy(), index=k["t"].to_numpy() + HOUR)  # index = close time
        try:
            funds[c] = load_funding(c, funding_src, s_ms, e_ms)
        except Exception as ex:
            print(f"  {c}: Funding nicht ladbar ({ex})")
            funds[c] = pd.DataFrame({"t": [], "rate": []})
        print(f"\r  geladen: {', '.join(closes)}", end="", flush=True)
    print()
    P = pd.DataFrame(closes).sort_index()
    idx = np.arange(P.index[0], P.index[-1] + HOUR, HOUR)
    P = P.reindex(idx).ffill()
    Fa = np.zeros(P.shape)
    for j, c in enumerate(P.columns):
        f = funds[c]
        if len(f):
            pos = np.searchsorted(idx, f["t"].to_numpy(), side="left")
            ok = pos < len(idx)
            np.add.at(Fa[:, j], pos[ok], f["rate"].to_numpy()[ok])
    return P, pd.DataFrame(Fa, index=idx, columns=P.columns)


# --------------------------------------------------------------------------- simulation
def simulate(P: pd.DataFrame, F: pd.DataFrame, i0: int, i1: int, cfg: RotCfg) -> dict:
    p_all = P.to_numpy()
    f_all = F.to_numpy()
    logp = np.log(p_all)
    L = cfg.lookback_h
    if cfg.vol_adj:
        r1 = np.diff(logp, axis=0, prepend=np.nan)
        vol = pd.DataFrame(r1).rolling(L, min_periods=L // 2).std().to_numpy() * np.sqrt(L)
    K = p_all.shape[1]
    cash, u, sides = cfg.pool, np.zeros(K), np.zeros(K)
    n_steps = i1 - i0
    eq = np.empty(n_steps)
    side_log = np.zeros((n_steps, K), dtype=np.int8)
    fees = funding = turnover = 0.0
    trades = 0
    for j, t in enumerate(range(i0, i1)):
        pv = np.nan_to_num(p_all[t])
        ft = f_all[t]
        if ft.any():
            pay = float(np.sum(u * pv * ft))
            cash -= pay
            funding -= pay
        s = t - cfg.exec_lag_h  # signal from an earlier close, trade at this close
        if j % cfg.rebalance_h == 0 and s - L >= 0:
            e = cash + float(np.sum(u * pv))
            sig = logp[s] - logp[s - L]
            if cfg.vol_adj:
                with np.errstate(invalid="ignore", divide="ignore"):
                    sig = sig / vol[s]
            valid = np.isfinite(sig) & (pv > 0)
            nv = int(valid.sum())
            if nv >= 2 * cfg.n:
                order = np.argsort(np.where(valid, sig, np.inf))[:nv]   # ascending, invalid dropped
                low, high = order[: cfg.n], order[-cfg.n:]
                new = np.zeros(K)
                new[high], new[low] = (1, -1) if cfg.mode == "momentum" else (-1, 1)
                notional = e * cfg.gross / (2 * cfg.n)
                target = np.where(pv > 0, new * notional / np.where(pv > 0, pv, 1), 0.0)
                du = target - u
                keep = (new == sides) & (new != 0) & (np.abs(du * pv) < cfg.band * notional)
                du[keep] = 0.0
                tv = np.abs(du * pv)
                cost = float(tv.sum()) * (cfg.fee + cfg.slippage)
                cash -= float(np.sum(du * pv)) + cost
                u += du
                fees += cost
                turnover += float(tv.sum())
                trades += int((tv > 0).sum())
                sides = new
        eq[j] = cash + float(np.sum(u * pv))
        side_log[j] = sides
    return dict(equity=eq, sides=side_log, fees=fees, funding=funding, turnover=turnover, trades=trades)


def metrics(res: dict, btc: np.ndarray, cfg: RotCfg) -> dict:
    eq = res["equity"]
    days = len(eq) / 24
    total = eq[-1] - cfg.pool
    r = np.diff(eq) / eq[:-1]
    rb = np.diff(btc) / btc[:-1]
    mid = len(eq) // 2
    dd = eq / np.maximum.accumulate(eq) - 1
    avg_eq = max(float(np.mean(eq)), 1e-9)
    return {
        "Netto $": total,
        "Netto %": 100 * total / cfg.pool,
        "APR %": 100 * total / cfg.pool / days * 365,
        "1. Haelfte $": eq[mid] - cfg.pool,
        "2. Haelfte $": eq[-1] - eq[mid],
        "Kurs-PnL $": total + res["fees"] - res["funding"],
        "Gebuehren+Slippage $": -res["fees"],
        "Funding $": res["funding"],
        "Max Drawdown %": 100 * dd.min(),
        "Sharpe": r.mean() / r.std() * np.sqrt(8760) if r.std() > 0 else 0.0,
        "Beta zu BTC": np.cov(r, rb)[0, 1] / np.var(rb),
        "Korrelation zu BTC": np.corrcoef(r, rb)[0, 1],
        "Umsatz $": res["turnover"],
        "Umsatz x Equity avg p.a.": res["turnover"] / avg_eq / days * 365,  # positions scale with equity
        "Trades": res["trades"],
    }


# --------------------------------------------------------------------------- report
def monthly(eq: np.ndarray, times: np.ndarray) -> pd.Series:
    s = pd.Series(eq, index=pd.to_datetime(times, unit="ms"))
    m = s.resample("ME").last()
    prev = m.shift(1)
    prev.iloc[0] = s.iloc[0]
    return 100 * (m / prev - 1)


def heatmap_fig(sweep: pd.DataFrame, periods: list[str]) -> go.Figure:
    lbs = sorted(sweep.lookback_h.unique())
    rbs = sorted(sweep.rebalance_h.unique())
    combos = [(p, m) for p in periods for m in ("momentum", "reversal")]
    fig = make_subplots(rows=1, cols=len(combos), shared_yaxes=True, horizontal_spacing=0.03,
                        subplot_titles=[f"{m}<br>{p}" for p, m in combos])
    for col, (p, m) in enumerate(combos, start=1):
        d = sweep[(sweep.period == p) & (sweep["mode"] == m)]
        z = d.pivot_table(index="lookback_h", columns="rebalance_h", values="Netto %", aggfunc="mean")
        z = z.reindex(index=lbs, columns=rbs)
        fig.add_trace(go.Heatmap(z=z.values, x=[f"RB {x}h" for x in rbs], y=[f"LB {y}h" for y in lbs],
                                 zmid=0, zmin=-40, zmax=40, colorscale="RdYlGn", showscale=col == 1,
                                 text=np.round(z.values, 1), texttemplate="%{text}%",
                                 hovertemplate="%{y} %{x}: %{z:.1f}%<extra></extra>"), row=1, col=col)
    fig.update_layout(height=420, template="plotly_dark",
                      title="Netto % je Lookback (LB) x Rebalancing (RB) - Mittel ueber N=2/3 und mit/ohne Vol-Adj.")
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coins", default=UNIVERSE)
    ap.add_argument("--months", type=int, default=12)
    ap.add_argument("--end", default="2026-11-06T10:00")
    ap.add_argument("--no-prev-year", "--no-oos", dest="no_prev", action="store_true",
                    help="keinen Vorjahres-Zeitraum testen")
    ap.add_argument("--funding", default="binance", help="binance | hyperliquid | none")
    ap.add_argument("--pool", type=float, default=2000)
    ap.add_argument("--lookback", type=int, default=168, help="Stunden")
    ap.add_argument("--rebalance", type=int, default=8, help="Stunden")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--gross", type=float, default=1.0)
    ap.add_argument("--no-vol-adj", action="store_true")
    ap.add_argument("--fee", type=float, default=0.045, help="Taker-Fee in %%")
    ap.add_argument("--slippage", type=float, default=0.03, help="in %%")
    ap.add_argument("--exec-lag", type=int, default=1,
                    help="Stunden zwischen Ranking-Schlusskurs und Ausfuehrung (0 = altes Verhalten)")
    ap.add_argument("--tag", help="Suffix fuer Ausgabedateien (Standard: rb<h>, wenn --rebalance nicht 8)")
    ap.add_argument("--report", default=None)
    a = ap.parse_args()
    tag = a.tag or (f"rb{a.rebalance}" if a.rebalance != 8 else "")
    sfx = f"_{tag}" if tag else ""
    a.report = a.report or f"rotation_report{sfx}.html"

    end = datetime.fromisoformat(a.end).replace(tzinfo=timezone.utc)
    span = timedelta(days=round(a.months * 30.44))
    periods = {"letzte %d Monate" % a.months: (end - span, end)}
    if not a.no_prev:
        # previous year: same data was used to choose the setup -> no out-of-sample test
        periods = {"Vorjahr": (end - 2 * span, end - span), **periods}
    load_start = min(s for s, _ in periods.values()) - timedelta(days=32)
    coins = [c.strip().upper() for c in a.coins.split(",")]
    print(f"Lade 1h-Kerzen + Funding ({a.funding}) fuer {len(coins)} Coins ab {load_start:%Y-%m-%d} ...")
    P, F = load_universe(coins, int(load_start.timestamp() * 1000), int(end.timestamp() * 1000), a.funding)
    times = P.index.to_numpy()
    idx = {k: (int(np.searchsorted(times, s.timestamp() * 1000)), int(np.searchsorted(times, e.timestamp() * 1000)))
           for k, (s, e) in periods.items()}

    base = RotCfg(lookback_h=a.lookback, rebalance_h=a.rebalance, n=a.n, gross=a.gross, vol_adj=not a.no_vol_adj,
                  fee=a.fee / 100, slippage=a.slippage / 100, pool=a.pool, exec_lag_h=a.exec_lag)
    variants = [base, replace(base, rebalance_h=72), replace(base, mode="reversal"),
                replace(base, fee=0.0, slippage=0.0)]
    names = ["Momentum (Standard)", "Momentum langsam (RB 72h)", "Gegenteil: Reversal", "Momentum ohne Gebuehren"]

    # ---------- main runs
    runs, rows = {}, []
    for per, (i0, i1) in idx.items():
        btc = P["BTC"].to_numpy()[i0:i1]
        for name, cfg in zip(names, variants):
            res = simulate(P, F, i0, i1, cfg)
            runs[(per, name)] = res
            rows.append({"Zeitraum": per, "Variante": name, **metrics(res, btc, cfg)})
        bh = a.pool * btc / btc[0]
        rows.append({"Zeitraum": per, "Variante": "BTC Buy & Hold", "Netto $": bh[-1] - a.pool,
                     "Netto %": 100 * (bh[-1] / a.pool - 1),
                     "Max Drawdown %": 100 * (bh / np.maximum.accumulate(bh) - 1).min()})
    tab = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(f"\nUniversum: {', '.join(P.columns)} | {base.label} | Gross {a.gross}x | Pool {a.pool:,.0f} $ | "
          f"Ausfuehrung {a.exec_lag} h nach dem Ranking")
    for per in idx:
        print(f"\n=== {per} ===")
        print(tab[tab.Zeitraum == per].drop(columns="Zeitraum").set_index("Variante").T
              .to_string(float_format=lambda x: f"{x:,.2f}"))

    # ---------- execution delay sensitivity
    lrow = []
    for per, (i0, i1) in idx.items():
        btc = P["BTC"].to_numpy()[i0:i1]
        for name, cfg in zip(names[:2], variants[:2]):
            for lag in (0, 1, 2, 4, 8, 12):
                m = metrics(simulate(P, F, i0, i1, replace(cfg, exec_lag_h=lag)), btc, cfg)
                lrow.append({"Zeitraum": per, "Variante": name, "Verzoegerung h": lag, "Netto %": m["Netto %"],
                             "Max Drawdown %": m["Max Drawdown %"]})
    ltab = pd.DataFrame(lrow)
    lpiv = ltab.pivot_table(index=["Variante", "Verzoegerung h"], columns="Zeitraum", values="Netto %", sort=False)
    print("\nVerzoegerung Ranking -> Ausfuehrung (Netto %):")
    print(lpiv.round(1).to_string())
    notes = [f"Ausfuehrung {a.exec_lag} h nach dem Ranking-Schlusskurs (--exec-lag 0 = altes Verhalten).",
             "Vorjahr ist kein Out-of-Sample-Test: Setup und Parameter wurden auf denselben Daten gewaehlt.",
             "Umsatz x Equity avg p.a. = gehandeltes Volumen / durchschnittliche Equity, auf 365 Tage skaliert."]
    print("\nHinweise:\n- " + "\n- ".join(notes))

    # ---------- sweep
    print("\nSweep laeuft ...")
    grid = list(itertools.product([24, 72, 168, 336, 720], [8, 24, 72], [2, 3], [True, False],
                                  ["momentum", "reversal"]))
    srows = []
    for per, (i0, i1) in idx.items():
        btc = P["BTC"].to_numpy()[i0:i1]
        for lb, rb, n, va, mode in grid:
            cfg = replace(base, lookback_h=lb, rebalance_h=rb, n=n, vol_adj=va, mode=mode)
            m = metrics(simulate(P, F, i0, i1, cfg), btc, cfg)
            srows.append(dict(period=per, lookback_h=lb, rebalance_h=rb, n=n, vol_adj=va, mode=mode, **m))
    sweep = pd.DataFrame(srows)
    out = report_dir("rotation")
    sweep.round(3).to_csv(out / f"rotation_sweep{sfx}.csv", index=False)
    tab.round(3).to_csv(out / f"rotation_results{sfx}.csv", index=False)
    ltab.round(3).to_csv(out / f"rotation_lag{sfx}.csv", index=False)
    for per in idx:
        d = sweep[sweep.period == per]
        for mode in ("momentum", "reversal"):
            dm = d[d["mode"] == mode]
            print(f"  {per:<26} {mode:<9}: {(dm['Netto $'] > 0).sum():>2}/{len(dm)} positiv, "
                  f"Median {dm['Netto %'].median():+6.1f} %, bestes {dm['Netto %'].max():+6.1f} %, "
                  f"schlechtestes {dm['Netto %'].min():+6.1f} %")
    if len(idx) == 2:
        p0, p1 = list(idx)
        piv = sweep.pivot_table(index=["mode", "lookback_h", "rebalance_h", "n", "vol_adj"], columns="period",
                                values="Netto %")
        corr = piv[p0].corr(piv[p1])
        top = piv.sort_values(p1, ascending=False).head(5)
        print(f"\n  Korrelation der Ergebnisse Vorjahr vs. letzte 12 Monate (alle Settings): {corr:+.2f}")
        print("  Top 5 der letzten 12 Monate und was sie im Vorjahr gebracht haetten:")
        print("  " + top.round(1).to_string().replace("\n", "\n  "))

    # ---------- report
    pers = list(idx)
    fig = make_subplots(rows=len(pers), cols=1, vertical_spacing=0.08, subplot_titles=[f"Equity - {p}" for p in pers])
    colors = {"Momentum (Standard)": "#00cc96", "Momentum langsam (RB 72h)": "#ffa15a",
              "Gegenteil: Reversal": "#ef553b", "Momentum ohne Gebuehren": "#636efa"}
    for row, per in enumerate(pers, start=1):
        i0, i1 = idx[per]
        ts = pd.to_datetime(times[i0:i1], unit="ms")
        for name in names:
            fig.add_trace(go.Scattergl(x=ts, y=runs[(per, name)]["equity"], name=name, legendgroup=name,
                                       showlegend=row == 1, line=dict(color=colors[name], width=1.5)), row=row, col=1)
        btc = P["BTC"].to_numpy()[i0:i1]
        fig.add_trace(go.Scattergl(x=ts, y=a.pool * btc / btc[0], name="BTC Buy & Hold", legendgroup="bh",
                                   showlegend=row == 1, line=dict(color="#888", dash="dot", width=1)), row=row, col=1)
    fig.update_layout(height=380 * len(pers), template="plotly_dark",
                      title=f"Momentum-Rotation - {base.label}, Gross {a.gross}x, Pool {a.pool:,.0f} $, "
                            f"Fee {a.fee}% + Slippage {a.slippage}%, Ausfuehrung +{a.exec_lag} h")

    mfig = go.Figure()
    for per in pers:
        i0, i1 = idx[per]
        mr = monthly(runs[(per, names[0])]["equity"], times[i0:i1])
        mfig.add_trace(go.Bar(x=mr.index.strftime("%Y-%m"), y=mr.values, name=per,
                              marker_color=["#00cc96" if v > 0 else "#ef553b" for v in mr.values],
                              text=[f"{v:+.1f}%" for v in mr.values], textposition="outside"))
    mfig.update_layout(height=380, template="plotly_dark", showlegend=False,
                       title="Monatsrenditen Momentum (Standard)")

    per = pers[-1]
    i0, i1 = idx[per]
    sl = runs[(per, names[0])]["sides"][:: base.rebalance_h]
    pfig = go.Figure(go.Heatmap(z=sl.T, x=pd.to_datetime(times[i0:i1:base.rebalance_h], unit="ms"),
                                y=list(P.columns), colorscale=[[0, "#ef553b"], [0.5, "#222"], [1, "#00cc96"]],
                                zmin=-1, zmax=1, showscale=False,
                                hovertemplate="%{y} %{x}: %{z}<extra>1=Long, -1=Short</extra>"))
    pfig.update_layout(height=420, template="plotly_dark",
                       title=f"Positionen Momentum (Standard) - {per}: gruen = Long, rot = Short")

    def table_html(df):
        return df.round(2).to_html(index=False, border=0, classes="t", na_rep="")

    css = ("<style>body{background:#111;color:#ddd;font-family:sans-serif}.t{border-collapse:collapse;margin:10px}"
           ".t td,.t th{padding:4px 10px;border-bottom:1px solid #333;text-align:right}</style>")
    html = ["<h3>Hinweise</h3><ul>" + "".join(f"<li>{n}</li>" for n in notes) + "</ul>",
            fig.to_html(full_html=False, include_plotlyjs="cdn"), mfig.to_html(full_html=False, include_plotlyjs=False),
            heatmap_fig(sweep, pers).to_html(full_html=False, include_plotlyjs=False),
            pfig.to_html(full_html=False, include_plotlyjs=False), "<h3>Kennzahlen</h3>", table_html(tab),
            "<h3>Verzoegerung Ranking -> Ausfuehrung (Netto %)</h3>", lpiv.round(1).to_html(border=0, classes="t")]
    (out / a.report).write_text(f"<html><head><meta charset='utf-8'>{css}</head><body>" + "".join(html)
                                + "</body></html>", encoding="utf-8")
    print(f"\nReport: {out / a.report}")


if __name__ == "__main__":
    main()
