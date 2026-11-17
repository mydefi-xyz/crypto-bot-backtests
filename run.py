"""Grid bot (liquidity range) backtest: standard long grid vs. symmetric (hedged) grids.

Example:
    python run.py                                  # BTC+ETH, last 180 days, defaults
    python run.py --coins BTC --range 5 --step 0.15
    python run.py --exchange hyperliquid --funding hyperliquid
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data import load_candles, load_funding, report_dir
from engine import Config, simulate

# maker %, taker % (base tier; override with --maker / --taker)
# extended: maker 0.000 %, taker 0.025 % for perps and spot per the official fee page
#   https://docs.extended.exchange/extended-resources/trading/trading-fees-and-rebates (checked 2026-11-17;
#   page states the schedule may change, makers can additionally earn volume-based rebates)
EXCHANGES = {
    "extended": (0.0, 0.025),
    "hyperliquid": (0.015, 0.045),
    "lighter": (0.0, 0.0),
    "binance": (0.02, 0.05),
}
MIN_QTY = {"BTC": 0.0001, "ETH": 0.001}
COLORS = {"1": "#ef553b", "2": "#00cc96", "3": "#636efa"}


def parse_time(s: str | None, default: datetime) -> datetime:
    if not s:
        return default
    d = datetime.fromisoformat(s)
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def build_configs(a, coin: str) -> list[Config]:
    maker, taker = EXCHANGES[a.exchange]
    maker = a.maker if a.maker is not None else maker
    taker = a.taker if a.taker is not None else taker
    common = dict(pool=a.pool, step=a.step / 100, range_pct=a.range / 100, pmin=a.pmin, pmax=a.pmax,
                  maker=maker / 100, taker=taker / 100, slippage=a.slippage / 100,
                  min_qty=MIN_QTY.get(coin, 0.0))
    protect = dict(stop_buffer=a.stop_buffer / 100, max_loss=a.max_loss / 100 if a.max_loss else None)
    restart = dict(restart=True, cooldown_min=int(a.cooldown * 60), calm_window=int(a.calm_hours * 60),
                   calm_range=a.calm / 100)
    all_cfgs = {
        "1": Config("1 Long-Grid (Standard)", "long", builder=a.builder / 100, **common),
        "2": Config("2 Symmetrisch + Stopp & Neustart", "sym", size_mult=a.size_mult,
                    **restart, **protect, **common),
        "3": Config("3 Symmetrisch + Nachziehen", "sym", size_mult=a.size_mult,
                    recenter_drift=a.recenter / 100 if a.recenter else None, **restart, **protect, **common),
    }
    return [all_cfgs[v] for v in a.variants.split(",")]


def chart(coin: str, df: pd.DataFrame, results, buyhold: np.ndarray) -> go.Figure:
    """One price panel per variant with a box per grid lifetime (green = profit, red = loss), plus equity."""
    ts = pd.to_datetime(df["t"], unit="ms")
    k = 15  # 15m resolution for the chart
    n = len(results)
    titles = [f"{coin} - {r.cfg.name}" for r in results] + ["Equity (USD)"]
    fig = make_subplots(rows=n + 1, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=[1] * n + [1.2], subplot_titles=titles)
    for row, r in enumerate(results, start=1):
        fig.add_trace(go.Scattergl(x=ts[::k], y=df["close"][::k], name=f"{coin} Kurs", showlegend=row == 1,
                                   line=dict(color="#bbb", width=1), hoverinfo="skip"), row=row, col=1)
        hx, hy, htxt, hcol = [], [], [], []
        for no, g in enumerate(r.grids, start=1):
            col = "#00cc96" if g["pnl"] > 0 else "#ef553b"
            x0, x1 = ts.iloc[g["start"]], ts.iloc[g["end"]]
            fig.add_shape(type="rect", x0=x0, x1=x1, y0=g["lo"], y1=g["hi"], row=row, col=1,
                          fillcolor=col, opacity=0.22, line=dict(color=col, width=1))
            dur = (g["end"] - g["start"]) / 1440
            hx.append(x0 + (x1 - x0) / 2)
            hy.append(g["hi"])
            hcol.append(col)
            htxt.append(f"<b>Grid #{no}</b> ({g['reason']})<br>{x0:%d.%m. %H:%M} -> {x1:%d.%m. %H:%M} "
                        f"({dur:.1f} Tage)<br>Range {g['lo']:,.0f} - {g['hi']:,.0f}<br>"
                        f"Fills {g['fills']}<br>Net fees {g['fees_net']:+.2f} $<br>"
                        f"Position {g['position']:+.2f} $<br>Funding {g['funding']:+.2f} $<br>"
                        f"<b>Netto {g['pnl']:+.2f} $</b>")
        fig.add_trace(go.Scatter(x=hx, y=hy, mode="markers", marker=dict(size=6, color=hcol, symbol="square"),
                                 hovertext=htxt, hoverinfo="text", showlegend=False), row=row, col=1)
    for r in results:
        fig.add_trace(go.Scattergl(x=ts[::k], y=r.equity[::k], name=r.cfg.name,
                                   line=dict(color=COLORS[r.cfg.name[0]], width=1.5)), row=n + 1, col=1)
    fig.add_trace(go.Scattergl(x=ts[::k], y=buyhold[::k], name="Buy & Hold (Referenz)",
                               line=dict(color="#888", width=1, dash="dot")), row=n + 1, col=1)
    fig.update_layout(height=320 * (n + 1), template="plotly_dark", hovermode="closest",
                      title=f"Grid-Bot Backtest - {coin}  (Kasten = ein Grid: gruen Gewinn, rot Verlust; "
                            f"Maus auf Quadrat = Details)", legend=dict(orientation="h", y=-0.05))
    return fig


def grid_summary(r) -> pd.DataFrame:
    g = pd.DataFrame(r.grids)
    if g.empty:
        return g
    return (g.groupby("reason")
             .agg(Anzahl=("pnl", "size"), Tage=("end", lambda s: (s - g.loc[s.index, "start"]).sum() / 1440),
                  Net_fees=("fees_net", "sum"), Position=("position", "sum"), Funding=("funding", "sum"),
                  Netto=("pnl", "sum"))
             .round(2))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--coins", default="BTC,ETH")
    p.add_argument("--days", type=float, default=180)
    p.add_argument("--start", help="ISO, z.B. 2026-11-01T20:05 (UTC)")
    p.add_argument("--end", help="ISO (UTC), default: jetzt")
    p.add_argument("--variants", default="1,2,3")
    p.add_argument("--pool", type=float, default=2000)
    p.add_argument("--step", type=float, default=0.30, help="Stufenabstand / fee rate in %%")
    p.add_argument("--range", type=float, default=10, help="halbe Range in %% (typisch: 5/10/20/30)")
    p.add_argument("--pmin", type=float, help="explizite Range (nur fuer erste Grid)")
    p.add_argument("--pmax", type=float)
    p.add_argument("--exchange", default="extended", choices=EXCHANGES)
    p.add_argument("--maker", type=float, help="Maker-Fee in %% (ueberschreibt Exchange-Preset)")
    p.add_argument("--taker", type=float, help="Taker-Fee in %%")
    p.add_argument("--builder", type=float, default=0.02, help="Anbieter-Gebuehr (Builder-Fee) pro Fill in %% (nur Variante 1)")
    p.add_argument("--funding", default="hyperliquid",
                   help="hyperliquid | lighter | binance | none | Zahl (= konstant %% pro Stunde)")
    p.add_argument("--slippage", type=float, default=0.05, help="Slippage Market/Stop in %%")
    p.add_argument("--stop-buffer", type=float, default=0.5, help="Stopp-Abstand ausserhalb Range in %%")
    p.add_argument("--max-loss", type=float, default=2.0, help="Verlustlimit pro Grid in %% vom Pool (0=aus)")
    p.add_argument("--cooldown", type=float, default=4, help="Mindest-Wartezeit nach Stopp in h (Var. 2+3)")
    p.add_argument("--calm", type=float, default=3.0,
                   help="Neustart erst wenn Hoch-Tief der letzten --calm-hours < x %% (Var. 2+3)")
    p.add_argument("--calm-hours", type=float, default=12, help="Fenster fuer Seitwaerts-Erkennung in h")
    p.add_argument("--recenter", type=float, default=50,
                   help="Nachziehen ab x %% der halben Range, wenn gebuehrenfinanziert (0=aus)")
    p.add_argument("--size-mult", type=float, default=1.0, help="Ordergroesse Sym-Varianten vs. Long-Grid")
    p.add_argument("--report", help="HTML-Dateiname (Standard: liquidity_report_<tag>.html)")
    p.add_argument("--tag", help="Suffix fuer Ausgabedateien (Standard: <coins>_<start>-<end>_v<variants>)")
    a = p.parse_args()

    end = parse_time(a.end, datetime.now(timezone.utc))
    start = parse_time(a.start, end - timedelta(days=a.days))
    s_ms, e_ms = int(start.timestamp() * 1000), int(end.timestamp() * 1000)
    tag = a.tag or (f"{a.coins.upper().replace(',', '-')}_{start:%Y%m%d}-{end:%Y%m%d}_v{a.variants.replace(',', '')}"
                    + ("" if a.exchange == "extended" else f"_{a.exchange}"))
    report = a.report or f"liquidity_report_{tag}.html"
    print(f"Zeitraum: {start:%Y-%m-%d %H:%M} -> {end:%Y-%m-%d %H:%M} UTC | Exchange-Fees: {a.exchange} "
          f"| Funding: {a.funding} | Tag: {tag}")

    out = report_dir("liquidity")
    html, tables, grid_rows = [], [], []
    for coin in a.coins.split(","):
        coin = coin.strip().upper()
        df = load_candles(coin, s_ms, e_ms)
        try:
            fund = load_funding(coin, a.funding, s_ms, e_ms)
        except Exception as ex:  # keep going without funding
            print(f"  Funding ({a.funding}) nicht ladbar: {ex} -> ohne Funding")
            fund = load_funding(coin, "none", s_ms, e_ms)
        if len(fund):
            hours = (e_ms - s_ms) / 3_600_000
            print(f"  {coin} Funding: Ø {100 * fund.rate.sum() / hours:.5f} %/h "
                  f"(~{100 * fund.rate.sum() / hours * 24 * 365:.1f} % p.a. fuer Longs)")

        results = [simulate(df, fund, cfg) for cfg in build_configs(a, coin)]
        tab = pd.DataFrame({r.cfg.name: r.metrics for r in results})
        c0, c1 = df["close"].iloc[0], df["close"].iloc[-1]
        print(f"\n=== {coin}: {c0:,.0f} -> {c1:,.0f} ({100 * (c1 / c0 - 1):+.1f} %), "
              f"Pool {a.pool:,.0f} $, Range ±{a.range} %, Step {a.step} % ===")
        with pd.option_context("display.width", 200):
            print(tab.to_string(float_format=lambda x: f"{x:,.5f}" if 0 < abs(x) < 0.1 else f"{x:,.2f}"))
        tab.insert(0, "coin", coin)
        tables.append(tab)
        for r in results:
            if len(r.grids) > 1:
                print(f"\n  Grids nach Ende-Grund - {r.cfg.name}:")
                print("  " + grid_summary(r).to_string().replace("\n", "\n  "))
            gdf = pd.DataFrame(r.grids)
            gdf.insert(0, "variant", r.cfg.name)
            gdf.insert(0, "coin", coin)
            gdf["start"] = pd.to_datetime(df["t"].iloc[gdf["start"]].to_numpy(), unit="ms")
            gdf["end"] = pd.to_datetime(df["t"].iloc[gdf["end"]].to_numpy(), unit="ms")
            grid_rows.append(gdf)

        buyhold = a.pool * df["close"].to_numpy() / c0
        fig = chart(coin, df, results, buyhold)
        html.append(fig.to_html(full_html=False, include_plotlyjs="cdn" if not html else False))

    res_csv, grid_csv = out / f"liquidity_results_{tag}.csv", out / f"liquidity_grids_{tag}.csv"
    pd.concat(tables).to_csv(res_csv)
    grids = pd.concat(grid_rows)
    num = grids.select_dtypes("number").columns
    grids[num] = grids[num].round(2)
    grids.to_csv(grid_csv, index=False)
    (out / report).write_text("<html><head><meta charset='utf-8'></head><body style='background:#111'>"
                              + "".join(html) + "</body></html>", encoding="utf-8")
    print(f"\nReport: {out / report}\nTabellen: {res_csv}, {grid_csv}")


if __name__ == "__main__":
    main()
