"""Quick test: close the symmetric grid early at X % of the half range and restart.

    python sweep_exit.py
"""
from __future__ import annotations

import itertools
from datetime import datetime, timedelta, timezone

import pandas as pd

from data import load_candles, load_funding, report_dir
from engine import Config, simulate

PERIODS = {"Nov25-Mai26": "2026-05-10T10:31", "Mai26-Nov26": "2026-11-06T10:31"}


def main():
    rows = []
    for label, end_s in PERIODS.items():
        end = datetime.fromisoformat(end_s).replace(tzinfo=timezone.utc)
        s_ms, e_ms = int((end - timedelta(days=180)).timestamp() * 1000), int(end.timestamp() * 1000)
        for coin in ("BTC", "ETH"):
            df, fund = load_candles(coin, s_ms, e_ms), load_funding(coin, "hyperliquid", s_ms, e_ms)
            for rng, ex, now in itertools.product([5, 10, 20], [0.2, 0.3, 0.5, None], [True, False]):
                if ex is None and now:
                    continue  # edge stop only: restart always waits for cooldown + calm filter
                # maker 0 % / taker 0.025 % = Extended base tier (see run.py)
                cfg = Config("x", "sym", pool=2000, step=0.003, range_pct=rng / 100, maker=0.0, taker=0.00025,
                             slippage=0.0005, stop_buffer=0.005, max_loss=None, restart=True,
                             cooldown_min=240, calm_window=48 * 60, calm_range=0.06,
                             exit_drift=ex, exit_restart_now=now)
                r = simulate(df, fund, cfg)
                m = r.metrics
                rows.append(dict(period=label, coin=coin, range=rng,
                                 exit="nur Stopp am Rand" if ex is None else f"{int(ex * 100)}%",
                                 # "sofort" = restart at the close of the exit minute;
                                 # "seitwaerts" = >= 4 h pause and 48-h high-low < 6 %
                                 restart="sofort" if now else "seitwaerts",
                                 netto=m["Netto-Ergebnis $"], fees=m["Net collected fees $"],
                                 position=m["Positions-PnL $"], funding=m["Funding $"], maxdd=m["Max Drawdown $"],
                                 schliessungen=sum(1 for g in r.grids if g["reason"] in ("exit", "stop"))))
    d = pd.DataFrame(rows)
    d.to_csv(report_dir("liquidity") / "sweep_exit.csv", index=False)
    pd.set_option("display.width", 220)
    piv = d.pivot_table(index=["range", "exit", "restart"], columns=["period", "coin"], values="netto").round(0)
    piv[("Summe", "")] = piv.sum(axis=1)
    print("Netto $ (Pool 2.000 $, je 6 Monate):")
    print(piv.to_string())
    print("\nAufteilung (Summe ueber beide Perioden und Coins):")
    print(d.groupby(["range", "exit", "restart"])[["fees", "position", "funding", "netto", "schliessungen"]].sum()
          .round(0).to_string())


if __name__ == "__main__":
    main()
