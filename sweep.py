"""Parameter sweep for the symmetric grid with stop & restart (variant 2) and recentering (variant 3).

Prints net result for the full period and separately for the 1st and 2nd half
(a setting that only works in one half is luck, not a strategy).

    python sweep.py
    python sweep.py --coins BTC --days 180
"""
from __future__ import annotations

import argparse
import itertools
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta, timezone

import pandas as pd

from data import load_candles, load_funding, report_dir
from engine import Config, simulate

_DATA: dict = {}


def _init(coins, s_ms, e_ms, funding):
    for c in coins:
        _DATA[c] = (load_candles(c, s_ms, e_ms), load_funding(c, funding, s_ms, e_ms))


def _run(job):
    coin, variant, rng, step, calm, calm_h = job
    df, fund = _DATA[coin]
    # maker 0 % / taker 0.025 % = Extended base tier (see run.py)
    cfg = Config(f"{variant}", "sym", pool=2000, step=step / 100, range_pct=rng / 100,
                 maker=0.0, taker=0.00025, slippage=0.0005, stop_buffer=0.005, max_loss=0.02,
                 restart=True, cooldown_min=240, calm_window=int(calm_h * 60), calm_range=calm / 100,
                 recenter_drift=0.5 if variant == 3 else None)
    r = simulate(df, fund, cfg)
    mid = len(r.equity) // 2
    eq = r.equity
    return dict(coin=coin, var=variant, range=rng, step=step,
                calm="nur 4h Pause" if calm >= 100 else f"{calm}%/{calm_h}h",
                netto=eq[-1] - 2000, h1=eq[mid] - 2000, h2=eq[-1] - eq[mid],
                fees=r.metrics["Net collected fees $"], position=r.metrics["Positions-PnL $"],
                maxdd=r.metrics["Max Drawdown $"], grids=len(r.grids),
                aktiv=r.metrics["Zeit aktiv in Range %"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--coins", default="BTC,ETH")
    p.add_argument("--days", type=float, default=180)
    p.add_argument("--end", default="2026-11-06T10:31")
    p.add_argument("--funding", default="hyperliquid")
    a = p.parse_args()
    end = datetime.fromisoformat(a.end).replace(tzinfo=timezone.utc)
    start = end - timedelta(days=a.days)
    s_ms, e_ms = int(start.timestamp() * 1000), int(end.timestamp() * 1000)
    coins = [c.strip().upper() for c in a.coins.split(",")]
    _init(coins, s_ms, e_ms, a.funding)  # download/cache once in main process

    # restart rules after a stop: always >= 4 h pause; (100, 1) = no calm filter on top
    calms = [(100, 1), (2, 12), (3, 12), (4, 24), (6, 48)]
    jobs = [(c, v, r, s, cm, ch) for c, v, r, s in itertools.product(coins, [2, 3], [3, 5, 7, 10], [0.2, 0.3, 0.5])
            for cm, ch in calms]
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1),
                             initializer=_init, initargs=(coins, s_ms, e_ms, a.funding)) as ex:
        rows = list(ex.map(_run, jobs))

    df = pd.DataFrame(rows)
    out = report_dir("liquidity") / f"sweep_{end:%Y%m%d}.csv"
    df.to_csv(out, index=False)
    print(f"Ergebnis: {out}")
    pd.set_option("display.width", 200)
    for coin in coins:
        d = df[df.coin == coin].sort_values("netto", ascending=False)
        print(f"\n=== {coin}: {len(d)} Kombinationen | positiv: {(d.netto > 0).sum()} | "
              f"in BEIDEN Haelften positiv: {((d.h1 > 0) & (d.h2 > 0)).sum()} | "
              f"Median netto {d.netto.median():+.1f} $ ===")
        print("Top 8:")
        print(d.head(8).round(1).to_string(index=False))
        print("Flop 3:")
        print(d.tail(3).round(1).to_string(index=False))
    print("\nMittelwert netto je Seitwaerts-Filter (alle Coins/Ranges/Steps):")
    print(df.groupby(["var", "calm"]).netto.mean().unstack(0).round(1).to_string())
    print("\nMittelwert netto je Range:")
    print(df.groupby(["var", "range"]).netto.mean().unstack(0).round(1).to_string())


if __name__ == "__main__":
    main()
