"""Momentum rotation (cross-sectional momentum) with a typical commercial reference setup, Binance daily data.

Reference setup:
  - ranks the 40 most liquid crypto perps by return over the last 14 closed days
  - top fifth long, bottom fifth short, equal size (dollar neutral)
  - book split into `hold` tranches (default 3), one tranche rebuilt per day
  - base leverage 2x = gross exposure / equity, deleverage to base above base+1
  - maker execution; ~18 bp costs per rotation incl. a bot-provider fee
  - ranking at the daily close, execution `--exec-lag` days later (default 1 = next daily close)

Universe includes coins that were delisted later on Binance (status SETTLING); liquidity = trailing
`--liq-days` quote volume on Binance. Positions in a delisted coin are closed at its last price.
Hyperliquid universe: a coin is only tradable from its first Hyperliquid record on (first daily candle or
first funding record), never before.

    python rotation_xs.py
    python rotation_xs.py --lev 1 --hold 7 --cost 0.05
    python rotation_xs.py --exec-lag 0     # old behaviour: rank and trade at the same close
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data import CACHE, SESSION, load_funding, load_klines, perp_universe, report_dir

DAY = 86_400_000
OUT = report_dir("rotation")
HL_INFO = "https://api.hyperliquid.xyz/info"
HL_META_CACHE = CACHE / "hl_meta_coins.json"
MAJORS20 = ["BTC", "ETH", "SOL", "XRP", "BNB", "DOGE", "ADA", "LINK", "AVAX", "SUI", "LTC", "BCH", "TRX", "DOT",
            "TON", "XLM", "HBAR", "NEAR", "UNI", "AAVE"]
SWEEP_LOOKBACKS, SWEEP_HOLDS, SWEEP_MARKETS = [3, 7, 14, 28, 56], [3, 5, 7, 14], [20, 40, 80]


@dataclass(frozen=True)
class PPCfg:
    n_markets: int = 40
    lookback: int = 14
    quantile: float = 0.2
    hold: int = 3
    lev: float = 2.0
    delever_add: float = 1.0     # deleverage when gross/equity > lev + this
    cost: float = 0.0009         # per traded notional and side (18 bp per open+close)
    mode: str = "momentum"       # momentum | reversal
    pool: float = 2000.0
    exec_lag: int = 1            # days between ranking close and execution close (0 = same close)


# --------------------------------------------------------------------------- data
def build_panel(start_ms: int, end_ms: int, only: set[str] | None = None):
    uni = perp_universe()
    if only is not None:
        uni = uni[uni.coin.isin(only)]
    closes, vols = {}, {}
    for k, row in enumerate(uni.itertuples(), start=1):
        df = load_klines(row.coin, "1d", start_ms, end_ms)
        print(f"\r  Tageskerzen: {k}/{len(uni)}", end="", flush=True)
        if df.empty:
            continue
        s = df.set_index("t")
        c, v = s["close"].copy(), s["quote_vol"].copy()
        if row.delist_ms:
            dead = s.index + DAY > row.delist_ms
            c[dead] = np.nan
            v[dead] = np.nan
        closes[row.coin], vols[row.coin] = c, v
    print()
    C = pd.DataFrame(closes).sort_index()
    C = C.reindex(np.arange(C.index[0], C.index[-1] + DAY, DAY))
    V = pd.DataFrame(vols).reindex(index=C.index, columns=C.columns)
    return C, V


def _hl_to_binance(name: str) -> str:
    return "1000" + name[1:] if name.startswith("k") and name[1:].isupper() else name  # kPEPE -> 1000PEPE


def _hl_funding_first() -> dict[str, int]:
    """First record per coin in the cached Hyperliquid funding files (upper bound for the listing date)."""
    first: dict[str, int] = {}
    for p in CACHE.glob("*_funding_hyperliquid_*.pkl"):
        f = pd.read_pickle(p)
        if len(f):
            c = p.name.split("_funding_hyperliquid_")[0]
            first[c] = min(first.get(c, 1 << 62), int(f["t"].min()))
    return first


def hyperliquid_listing() -> tuple[dict[str, int | None], str]:
    """Hyperliquid coins (Binance base names, incl. delisted) -> first known Hyperliquid timestamp in ms.

    Order: live `meta` + first daily candle per coin (written to data/hl_meta_coins.json), then that cache
    file, then the names of the cached Hyperliquid funding files (incomplete list, source 'funding-cache').
    """
    fund_first = _hl_funding_first()
    cached = json.loads(HL_META_CACHE.read_text()) if HL_META_CACHE.exists() else None
    try:
        r = SESSION.post(HL_INFO, json={"type": "meta"}, timeout=20)
        r.raise_for_status()
        names = [x["name"] for x in r.json()["universe"]]
    except Exception as ex:  # API down -> cached data
        print(f"  Hyperliquid meta nicht erreichbar ({type(ex).__name__}), nutze Cache")
        names = None
    if names is not None:
        known = (cached or {}).get("coins", {})
        coins: dict[str, int | None] = {}
        now = int(time.time() * 1000)
        for k, n in enumerate(names, start=1):
            c = _hl_to_binance(n)
            first = known.get(c)
            if first is None:
                print(f"\r  Hyperliquid Listing-Daten: {k}/{len(names)} ({n})        ", end="", flush=True)
                try:
                    cs = SESSION.post(HL_INFO, json={"type": "candleSnapshot", "req": {
                        "coin": n, "interval": "1d", "startTime": 0, "endTime": now}}, timeout=30).json()
                    first = int(cs[0]["t"]) if cs else None
                except Exception:
                    first = None
                time.sleep(0.2)
            coins[c] = first
        print()
        HL_META_CACHE.write_text(json.dumps({"fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                             "source": "hyperliquid meta + first 1d candle", "coins": coins},
                                            indent=1), encoding="utf-8")
        src = "meta"
    elif cached is not None:
        coins, src = dict(cached["coins"]), f"meta-cache {cached.get('fetched_utc', '')}"
    else:
        return dict(fund_first), "funding-cache"
    for c, t in fund_first.items():  # a funding record also proves the listing
        if c in coins:
            coins[c] = t if coins[c] is None else min(coins[c], t)
    return coins, src


def listing_mask(C: pd.DataFrame, listing: dict[str, int | None] | None) -> pd.DataFrame:
    """True where a coin has a price and is listed on the venue (day t counts if listed before its close)."""
    A = C.notna()
    if listing is not None:
        for c in A.columns:
            first = listing.get(c)  # None = listing date unknown -> never tradable
            A[c] = A[c] & (A.index + DAY > first) if first is not None else False
    return A


def load_daily_funding(C: pd.DataFrame, coins, start_ms, end_ms) -> pd.DataFrame:
    F = pd.DataFrame(0.0, index=C.index, columns=C.columns)
    for k, c in enumerate(coins, start=1):
        print(f"\r  Funding: {k}/{len(coins)} ({c})        ", end="", flush=True)
        f = load_funding(c, "binance", start_ms, end_ms)
        if len(f):
            day = (f["t"] // DAY) * DAY
            F[c] = f.groupby(day)["rate"].sum().reindex(C.index).fillna(0.0).to_numpy()
    print()
    return F


def candidates(C: pd.DataFrame, liq: pd.DataFrame, A: pd.DataFrame, n_max: int, lookback_min: int) -> list[str]:
    """Coins that can ever be selected (same eligibility as simulate, any run/sweep) -> need funding data."""
    ok = A & C.shift(lookback_min).notna() & liq.notna()
    rank = liq.where(ok).rank(axis=1, ascending=False)
    return list(C.columns[(rank <= n_max).any()])


# --------------------------------------------------------------------------- simulation
def simulate(C, V, F, d0: int, d1: int, cfg: PPCfg, liq=None, avail=None) -> dict:
    Cv, Fv = C.to_numpy(), F.to_numpy()
    liq = (V.rolling(30, min_periods=20).mean() if liq is None else liq).to_numpy()
    av = np.isfinite(Cv) if avail is None else avail.to_numpy()
    K, H, L, lag = Cv.shape[1], cfg.hold, cfg.lookback, cfg.exec_lag
    U = np.zeros((H, K))
    cash = cfg.pool
    last = np.full(K, np.nan)
    eq, gross = np.empty(d1 - d0), np.empty(d1 - d0)
    coin_pnl, held_days = np.zeros(K), np.zeros(K)
    costs = funding = turnover = 0.0
    delevers = 0
    for j, d in enumerate(range(d0, d1)):
        px = Cv[d]
        okp = np.isfinite(px)
        u = U.sum(0)
        # pnl attribution + funding for the day just ended (positions from previous rebuild)
        held = u != 0
        if held.any():
            move = np.where(okp, px, last) - last
            coin_pnl[held] += u[held] * move[held]
            pay = float(np.nansum(u * np.where(okp, px, last) * Fv[d]))
            cash -= pay
            funding -= pay
            coin_pnl -= np.nan_to_num(u * np.where(okp, px, last) * Fv[d])
        # delisted coins: close at last known price
        dead = held & ~okp
        if dead.any():
            tv = float(np.abs(u[dead] * last[dead]).sum())
            cash += float((u[dead] * last[dead]).sum()) - tv * cfg.cost
            costs += tv * cfg.cost
            turnover += tv
            U[:, dead] = 0.0
        last[okp] = px[okp]
        mpx = np.where(okp, px, 0.0)
        u = U.sum(0)
        e = cash + float(u @ mpx)
        if e <= 0:  # account wiped out
            eq[j:], gross[j:] = 0.0, 0.0
            break
        # deleverage guard
        g = float(np.abs(u * mpx).sum()) / e
        if g > cfg.lev + cfg.delever_add:
            scale = cfg.lev / g
            du = u * (scale - 1)
            tv = float(np.abs(du * mpx).sum())
            cash -= float(du @ mpx) + tv * cfg.cost
            costs += tv * cfg.cost
            turnover += tv
            U *= scale
            delevers += 1
            e = cash + float(U.sum(0) @ mpx)
        # rebuild one tranche: ranking at the close of day s, execution at today's close
        s = d - lag
        if s - L >= 0:
            ret = Cv[s] / Cv[s - L] - 1
            elig = av[s] & np.isfinite(ret) & np.isfinite(liq[s])
            idx = np.flatnonzero(elig)
            if len(idx) >= 10:
                idx = idx[np.argsort(-liq[s, idx])][: cfg.n_markets]
                nq = max(1, int(round(cfg.quantile * len(idx))))
                order = idx[np.argsort(ret[idx])]
                winners, losers = order[-nq:], order[:nq]
                longs, shorts = (winners, losers) if cfg.mode == "momentum" else (losers, winners)
                longs, shorts = longs[okp[longs]], shorts[okp[shorts]]  # delisted since ranking -> skip
                notional = e * cfg.lev / H / 2 / nq
                old = U.sum(0)
                tr = (d - d0) % H
                U[tr] = 0.0
                U[tr, longs] = notional / px[longs]
                U[tr, shorts] = -notional / px[shorts]
                du = U.sum(0) - old
                tv = float(np.abs(du * mpx).sum())
                cash -= float(du @ mpx) + tv * cfg.cost
                costs += tv * cfg.cost
                turnover += tv
        u = U.sum(0)
        held_days += u != 0
        eq[j] = cash + float(u @ mpx)
        gross[j] = float(np.abs(u * mpx).sum()) / max(eq[j], 1e-9)
    return dict(equity=eq, gross=gross, costs=costs, funding=funding, turnover=turnover,
                delevers=delevers, coin_pnl=pd.Series(coin_pnl, index=C.columns),
                held_days=pd.Series(held_days, index=C.columns))


def metrics(res: dict, btc: np.ndarray, cfg: PPCfg) -> dict:
    eq = res["equity"]
    days = len(eq)
    total = eq[-1] - cfg.pool
    r = np.diff(eq, prepend=cfg.pool) / np.maximum(np.concatenate([[cfg.pool], eq[:-1]]), 1e-9)
    rb = np.diff(btc, prepend=btc[0]) / np.concatenate([[btc[0]], btc[:-1]])
    dd = eq / np.maximum.accumulate(np.maximum(eq, cfg.pool)) - 1
    mid = days // 2
    avg_eq = max(float(np.mean(eq)), 1e-9)
    return {
        "Netto $": total,
        "Netto %": 100 * total / cfg.pool,
        "p.a. %": 100 * ((eq[-1] / cfg.pool) ** (365 / days) - 1) if eq[-1] > 0 else -100.0,
        "1. Haelfte %": 100 * (eq[mid] / cfg.pool - 1),
        "2. Haelfte %": 100 * (eq[-1] / eq[mid] - 1) if eq[mid] > 0 else np.nan,
        "Max Drawdown %": 100 * dd.min(),
        "Sharpe": r.mean() / r.std() * np.sqrt(365) if r.std() > 0 else 0.0,
        "Beta zu BTC": np.cov(r, rb)[0, 1] / np.var(rb),
        "Kurs-PnL $": total + res["costs"] - res["funding"],
        "Kosten $": -res["costs"],
        "Funding $": res["funding"],
        "Umsatz $": res["turnover"],
        "Equity avg $": avg_eq,
        "Umsatz x Equity avg p.a.": res["turnover"] / avg_eq * 365 / days,  # positions compound with equity
        "Delever-Events": res["delevers"],
        "Hebel avg": float(np.mean(res["gross"])),
    }


def monthly(eq, times, pool):
    s = pd.Series(eq, index=pd.to_datetime(times, unit="ms"))
    m = s.resample("ME").last()
    prev = m.shift(1)
    prev.iloc[0] = pool
    return 100 * (m / prev - 1)


# --------------------------------------------------------------------------- main
def main():
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # some Binance coins have non-ASCII names
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--end", default="2026-11-06")
    ap.add_argument("--lev", type=float, default=2.0)
    ap.add_argument("--hold", type=int, default=3)
    ap.add_argument("--lookback", type=int, default=14)
    ap.add_argument("--markets", type=int, default=40)
    ap.add_argument("--cost", type=float, default=0.09, help="Kosten je gehandeltem Volumen und Seite in %%")
    ap.add_argument("--pool", type=float, default=2000)
    ap.add_argument("--universe", default="binance", choices=["binance", "hyperliquid"],
                    help="binance = alle Binance-Perps, hyperliquid = nur Coins, die auf Hyperliquid gelistet sind/waren")
    ap.add_argument("--no-funding", action="store_true", help="Funding ignorieren (Funding ist boersenabhaengig)")
    ap.add_argument("--coins", help="feste Coin-Liste, z.B. BTC,ETH,SOL (statt Top-N nach Volumen)")
    ap.add_argument("--liq-days", type=int, default=30, help="Fenster fuer Volumen-Ranking in Tagen")
    ap.add_argument("--exec-lag", type=int, default=1,
                    help="Tage zwischen Ranking-Schlusskurs und Ausfuehrung (Standard 1 = naechster Schlusskurs, "
                         "0 = altes Verhalten: Ranking und Ausfuehrung zum selben Schlusskurs)")
    ap.add_argument("--tag", help="Name fuer Ausgabedateien")
    ap.add_argument("--report", default=None)
    ap.add_argument("--skip-sweep", action="store_true")
    a = ap.parse_args()
    tag = a.tag or (a.universe + ("_nofunding" if a.no_funding else ""))
    a.report = a.report or f"rotation_xs_report_{tag}.html"

    end = datetime.fromisoformat(a.end).replace(tzinfo=timezone.utc)
    end_ms = int(end.timestamp() * 1000) // DAY * DAY
    y1, y2 = end_ms - 365 * DAY, end_ms - 730 * DAY
    load_start = y2 - 70 * DAY
    periods = {
        "Vorjahr": (y2, y1),
        "Letzte 12 Monate": (y1, end_ms),
        "Zeitraum Jan25-Okt26": (int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp() * 1000),
                                          int(datetime(2026, 11, 1, tzinfo=timezone.utc).timestamp() * 1000)),
        "Gesamt 24 Monate": (y2, end_ms),
    }
    wanted = [c.strip().upper() for c in a.coins.split(",") if c.strip()] if a.coins else None
    listing = None
    if a.universe == "hyperliquid":
        listing, src = hyperliquid_listing()
        print(f"  Hyperliquid-Coins: {len(listing)} (Quelle: {src})")
        if src == "funding-cache":
            # coin list derived from cached funding files is incomplete -> only fixed lists fully covered by it
            missing = [c for c in wanted or [] if c not in listing]
            if wanted is None or missing:
                raise SystemExit("Hyperliquid meta nicht erreichbar und kein data/hl_meta_coins.json vorhanden. "
                                 "Die Coin-Liste aus den Funding-Dateien ist unvollstaendig"
                                 + (f" (fehlt: {', '.join(missing)})" if missing else "")
                                 + ". Lauf spaeter wiederholen oder --coins mit gecachten Coins nutzen.")
    print(f"Lade Universum ({a.universe}, inkl. delisteter Coins) ...")
    C, V = build_panel(load_start, end_ms, set(listing) if listing is not None else None)
    if wanted:
        keep = [c for c in wanted if c in C.columns]
        if len(keep) < len(wanted):
            print(f"  nicht im Universum/ohne Daten: {', '.join(c for c in wanted if c not in keep)}")
        C, V = C[keep], V[keep]
        a.markets = min(a.markets, len(keep))
        print(f"  feste Liste: {', '.join(keep)}")
    A = listing_mask(C, listing)
    if listing is not None:
        late = [c for c in C.columns if (C[c].notna() & ~A[c]).iloc[np.searchsorted(C.index, y2):].any()]
        print(f"  {len(late)} Coins erst nach Testbeginn auf Hyperliquid handelbar (vorher gesperrt)")
    liq = V.rolling(a.liq_days, min_periods=min(20, a.liq_days)).mean()
    n_max = a.markets if a.skip_sweep else max(a.markets, *SWEEP_MARKETS)
    lb_min = a.lookback if a.skip_sweep else min(a.lookback, *SWEEP_LOOKBACKS)
    cand = candidates(C, liq, A, n_max, lb_min)
    print(f"  {C.shape[1]} Coins mit Daten, {len(cand)} davon jemals unter den Top {n_max} "
          f"({a.liq_days}-Tage-Volumen) -> Funding laden")
    F = (pd.DataFrame(0.0, index=C.index, columns=C.columns) if a.no_funding
         else load_daily_funding(C, cand, load_start, end_ms))
    times = C.index.to_numpy()
    di = {k: (int(np.searchsorted(times, s)), int(np.searchsorted(times, e))) for k, (s, e) in periods.items()}
    btc = C["BTC"].to_numpy()

    def sim(i0, i1, cfg):
        return simulate(C, V, F, i0, i1, cfg, liq, A)

    base = PPCfg(n_markets=a.markets, lookback=a.lookback, hold=a.hold, lev=a.lev, cost=a.cost / 100, pool=a.pool,
                 exec_lag=a.exec_lag)
    variants = {
        "Standard": base,
        "Hold 7": replace(base, hold=7),
        "Hebel 1x": replace(base, lev=1.0),
        "Gegenteil (Reversal)": replace(base, mode="reversal"),
        "Ohne Kosten": replace(base, cost=0.0),
    }
    runs, rows = {}, []
    for per, (i0, i1) in di.items():
        for name, cfg in variants.items():
            res = sim(i0, i1, cfg)
            runs[(per, name)] = res
            rows.append({"Zeitraum": per, "Variante": name, **metrics(res, btc[i0:i1], cfg)})
        b = btc[i0:i1]
        rows.append({"Zeitraum": per, "Variante": "BTC Buy & Hold", "Netto %": 100 * (b[-1] / b[0] - 1),
                     "Max Drawdown %": 100 * (b / np.maximum.accumulate(b) - 1).min()})
    tab = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(f"\nSetup: {a.markets} Maerkte, {a.lookback}d Return, Top/Flop 20 %, Hold {a.hold}, Hebel {a.lev}x, "
          f"Kosten {a.cost} % je Seite, Ausfuehrung {a.exec_lag} Tag(e) nach dem Ranking")
    for per in periods:
        print(f"\n=== {per} ===")
        print(tab[tab.Zeitraum == per].drop(columns="Zeitraum").set_index("Variante").T
              .to_string(float_format=lambda x: f"{x:,.2f}"))

    # cost sensitivity
    crow = []
    for per in ("Vorjahr", "Letzte 12 Monate", "Zeitraum Jan25-Okt26"):
        i0, i1 = di[per]
        for c in (0.0, 0.03, 0.06, 0.09, 0.15):
            m = metrics(sim(i0, i1, replace(base, cost=c / 100)), btc[i0:i1], base)
            crow.append({"Zeitraum": per, "Kosten % je Seite": c, "Netto %": m["Netto %"], "p.a. %": m["p.a. %"],
                         "Max DD %": m["Max Drawdown %"], "Sharpe": m["Sharpe"]})
    ctab = pd.DataFrame(crow)
    print("\nKosten-Sensitivitaet (Standard):")
    print(ctab.pivot(index="Kosten % je Seite", columns="Zeitraum", values="p.a. %").round(1).to_string())

    # execution delay sensitivity
    lrow = []
    for per, (i0, i1) in di.items():
        for name in ("Standard", "Hebel 1x"):
            for lag in (0, 1, 2):
                cfg = replace(variants[name], exec_lag=lag)
                m = metrics(sim(i0, i1, cfg), btc[i0:i1], cfg)
                lrow.append({"Zeitraum": per, "Variante": name, "Verzoegerung Tage": lag, "Netto %": m["Netto %"],
                             "Max DD %": m["Max Drawdown %"]})
    ltab = pd.DataFrame(lrow)
    ltab.round(3).to_csv(OUT / f"rotation_xs_lag_{tag}.csv", index=False)
    lpiv = ltab.pivot_table(index=["Variante", "Verzoegerung Tage"], columns="Zeitraum", values="Netto %",
                            sort=False)
    print("\nVerzoegerung Ranking -> Ausfuehrung (Netto %):")
    print(lpiv.round(1).to_string())

    # sweep lookback x hold x markets
    sweep = pd.DataFrame()
    if not a.skip_sweep:
        srows = []
        for per in ("Vorjahr", "Letzte 12 Monate"):
            i0, i1 = di[per]
            for lb, h, nm in itertools.product(SWEEP_LOOKBACKS, SWEEP_HOLDS, SWEEP_MARKETS):
                cfg = replace(base, lookback=lb, hold=h, n_markets=nm)
                m = metrics(sim(i0, i1, cfg), btc[i0:i1], cfg)
                srows.append(dict(period=per, lookback=lb, hold=h, markets=nm, **m))
        sweep = pd.DataFrame(srows)
        sweep.round(3).to_csv(OUT / f"rotation_xs_sweep_{tag}.csv", index=False)
        for per in ("Vorjahr", "Letzte 12 Monate"):
            d = sweep[sweep.period == per]
            print(f"  Sweep {per}: {(d['Netto %'] > 0).sum()}/{len(d)} positiv, Median {d['Netto %'].median():+.1f} %")
        piv = sweep.pivot_table(index=["lookback", "hold", "markets"], columns="period", values="Netto %")
        print(f"  Korrelation Vorjahr vs. letzte 12 Monate: {piv['Vorjahr'].corr(piv['Letzte 12 Monate']):+.2f}, "
              f"in beiden positiv: {((piv['Vorjahr'] > 0) & (piv['Letzte 12 Monate'] > 0)).sum()}/{len(piv)}")
    tab.round(3).to_csv(OUT / f"rotation_xs_results_{tag}.csv", index=False)

    # coin attribution (last 12 months, standard)
    cp = runs[("Letzte 12 Monate", "Standard")]["coin_pnl"]
    cp = cp[cp != 0].sort_values()
    print("\nGroesste Gewinner/Verlierer-Coins (letzte 12 Monate, $):")
    print(pd.concat([cp.head(8), cp.tail(8)]).round(0).to_string())

    # which coins were in the universe / held (24 months, standard)
    i0, i1 = di["Gesamt 24 Monate"]
    ok = A & C.shift(a.lookback).notna() & liq.notna()
    in_top = (liq.where(ok).rank(axis=1, ascending=False) <= a.markets).iloc[i0:i1]
    hd = runs[("Gesamt 24 Monate", "Standard")]["held_days"]
    hd = hd[hd > 0].sort_values(ascending=False)
    pos_days = float(hd.sum())
    major_share = 100 * hd[hd.index.isin(MAJORS20)].sum() / pos_days if pos_days else 0.0
    coin_stats = {"Coins jemals in Top N": int(in_top.any().sum()), "Coins immer in Top N": int(in_top.all().sum()),
                  "Coins gehalten": len(hd), "Anteil Positionstage 20 grosse Coins %": round(major_share, 1),
                  "BTC gehalten Tage": int(hd.get("BTC", 0)), "Tage gesamt": i1 - i0,
                  "meistgehalten": ", ".join(hd.index[:8])}
    hd.rename("Positionstage").to_csv(OUT / f"rotation_xs_coins_{tag}.csv")
    print("\nCoins (24 Monate, Standard): " + " | ".join(f"{k}: {v}" for k, v in coin_stats.items()))

    notes = [
        f"Ausfuehrung {a.exec_lag} Tag(e) nach dem Ranking-Schlusskurs (--exec-lag 0 = altes Verhalten).",
        "Vorjahr ist kein Out-of-Sample-Test: Setup, Coin-Listen und Volumen-Fenster wurden auf denselben "
        "24 Monaten gewaehlt.",
        "Umsatz x Equity avg p.a. = gehandeltes Volumen / durchschnittliche Equity, auf 365 Tage skaliert "
        "(Positionen wachsen mit dem Konto).",
        "Delistete Coins: nur Binance-Perps mit Status SETTLING sind enthalten; Symbole, die Binance ganz "
        "entfernt hat, fehlen (Rest-Survivorship).",
    ]
    if listing is not None:
        notes.append(f"Hyperliquid-Universum ({src}): Coin erst ab erstem Hyperliquid-Datensatz handelbar.")
    print("\nHinweise:\n- " + "\n- ".join(notes))

    # ------------------------------------------------------------------ report
    colors = {"Standard": "#00cc96", "Hold 7": "#ffa15a", "Hebel 1x": "#ab63fa",
              "Gegenteil (Reversal)": "#ef553b", "Ohne Kosten": "#636efa"}
    pers = ["Vorjahr", "Letzte 12 Monate", "Zeitraum Jan25-Okt26"]
    fig = make_subplots(rows=3, cols=1, vertical_spacing=0.07, subplot_titles=[f"Equity - {p}" for p in pers])
    for row, per in enumerate(pers, start=1):
        i0, i1 = di[per]
        ts = pd.to_datetime(times[i0:i1], unit="ms")
        for name in variants:
            fig.add_trace(go.Scatter(x=ts, y=runs[(per, name)]["equity"], name=name, legendgroup=name,
                                     showlegend=row == 1, line=dict(color=colors[name], width=1.5)), row=row, col=1)
        b = btc[i0:i1]
        fig.add_trace(go.Scatter(x=ts, y=a.pool * b / b[0], name="BTC Buy & Hold", legendgroup="bh",
                                 showlegend=row == 1, line=dict(color="#888", dash="dot", width=1)), row=row, col=1)
    fig.update_layout(height=1100, template="plotly_dark",
                      title=f"Momentum-Rotation (Referenz-Setup) - {a.markets} Maerkte, {a.lookback}d, Hold {a.hold}, "
                            f"{a.lev}x, Kosten {a.cost}% je Seite, Ausfuehrung +{a.exec_lag} Tag(e)")

    i0, i1 = di["Gesamt 24 Monate"]
    mr = monthly(runs[("Gesamt 24 Monate", "Standard")]["equity"], times[i0:i1], a.pool)
    mfig = go.Figure(go.Bar(x=mr.index.strftime("%Y-%m"), y=mr.values, text=[f"{v:+.1f}%" for v in mr.values],
                            textposition="outside", marker_color=["#00cc96" if v > 0 else "#ef553b" for v in mr.values]))
    mfig.update_layout(height=380, template="plotly_dark", title="Monatsrenditen Standard (24 Monate)")

    hfig = make_subplots(rows=1, cols=2, subplot_titles=["Vorjahr", "Letzte 12 Monate"], shared_yaxes=True)
    for col, per in enumerate(("Vorjahr", "Letzte 12 Monate") if len(sweep) else (), start=1):
        d = sweep[(sweep.period == per) & (sweep.markets == a.markets)]
        z = d.pivot_table(index="lookback", columns="hold", values="Netto %")
        hfig.add_trace(go.Heatmap(z=z.values, x=[f"Hold {h}" for h in z.columns], y=[f"{l}d" for l in z.index],
                                  zmid=0, zmin=-60, zmax=60, colorscale="RdYlGn", showscale=col == 1,
                                  text=np.round(z.values, 1), texttemplate="%{text}%"), row=1, col=col)
    hfig.update_layout(height=380, template="plotly_dark",
                       title=f"Netto % je Return-Fenster x Hold ({a.markets} Maerkte, {a.lev}x)")

    afig = go.Figure(go.Bar(x=pd.concat([cp.head(12), cp.tail(12)]).values,
                            y=pd.concat([cp.head(12), cp.tail(12)]).index, orientation="h",
                            marker_color=["#ef553b"] * min(12, len(cp)) + ["#00cc96"] * min(12, len(cp))))
    afig.update_layout(height=600, template="plotly_dark", title="PnL je Coin - letzte 12 Monate ($)")

    css = ("<style>body{background:#111;color:#ddd;font-family:sans-serif}.t{border-collapse:collapse;margin:10px}"
           ".t td,.t th{padding:4px 10px;border-bottom:1px solid #333;text-align:right}</style>")
    parts = ["<h3>Hinweise</h3><ul>" + "".join(f"<li>{n}</li>" for n in notes) + "</ul>",
             fig.to_html(full_html=False, include_plotlyjs="cdn"), mfig.to_html(full_html=False, include_plotlyjs=False),
             hfig.to_html(full_html=False, include_plotlyjs=False), afig.to_html(full_html=False, include_plotlyjs=False),
             "<h3>Kennzahlen</h3>", tab.round(2).to_html(index=False, border=0, classes="t", na_rep=""),
             "<h3>Kosten-Sensitivitaet (p.a. %)</h3>",
             ctab.pivot(index="Kosten % je Seite", columns="Zeitraum", values="p.a. %").round(1)
             .to_html(border=0, classes="t"),
             "<h3>Verzoegerung Ranking -> Ausfuehrung (Netto %)</h3>", lpiv.round(1).to_html(border=0, classes="t"),
             "<h3>Coins (24 Monate, Standard)</h3>",
             pd.Series(coin_stats).to_frame("Wert").to_html(border=0, classes="t")]
    (OUT / a.report).write_text(f"<html><head><meta charset='utf-8'>{css}</head><body>" + "".join(parts)
                                + "</body></html>", encoding="utf-8")
    print(f"\nReport: {OUT / a.report}")


if __name__ == "__main__":
    main()
