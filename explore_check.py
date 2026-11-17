"""Nachrechnung typischer APR-Ranglisten fuer Grid-Setups: gleiche Range-Breite, Fee-Rate und 30 Tage,
nachgefuehrte Range (bei Austritt neu zentriert) - aber MIT Kursbewegung des Bestands.

    python explore_check.py
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone

import pandas as pd

from data import load_candles, load_funding, report_dir
from engine import Config, simulate

# coin: (Range low, Range high, Fee rate %, maker %, taker %, angezeigte APR %, angezeigte Fee-APR %)
SETUPS = {
    "ZEC": (1202, 1435, 0.30, 0.0144, 0.0432, 307.0, 318.7),
    "HYPE": (75.56, 82.60, 0.15, 0.0144, 0.0432, 278.0, None),
    "ETH": (2096, 2615, 0.80, 0.012, 0.035, 35.9, 40.1),
    "BNB": (672.5, 833.6, 0.50, 0.0144, 0.0432, 28.9, None),
    "XRP": (1.15, 1.58, 0.50, 0.0, 0.0, 25.8, None),
    "BTC": (74067, 88843, 0.50, 0.012, 0.035, 17.3, None),
}


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    end = datetime(2026, 11, 8, 19, 40, tzinfo=timezone.utc)
    start = end - timedelta(days=30)
    s_ms, e_ms = int(start.timestamp() * 1000), int(end.timestamp() * 1000)
    rows = []
    for coin, (lo, hi, step, mk, tk, apr_shown, fee_shown) in SETUPS.items():
        h = (hi - lo) / (hi + lo)                     # halbe Breite relativ zur Mitte
        df = load_candles(coin, s_ms, e_ms)
        try:
            fund = load_funding(coin, "hyperliquid", s_ms, e_ms)
        except Exception as ex:  # noqa: BLE001
            print(f"  {coin}: Hyperliquid-Funding nicht ladbar ({ex}) -> ohne Funding")
            fund = load_funding(coin, "none", s_ms, e_ms)
        vol = df["close"].pct_change().std() * (525_600 ** 0.5)
        for mode, label in (("long", "Long-Grid"), ("sym", "Symmetrisch")):
            cfg = Config(label, mode, pool=10_000, step=step / 100, range_pct=h, maker=mk / 100, taker=tk / 100,
                         slippage=0.0005, exit_drift=1.0, exit_restart_now=True)
            m = simulate(df, fund, cfg).metrics
            k = 100 / cfg.pool * 365 / 30
            rows.append({"Coin": coin, "Variante": label, "Range ±%": round(100 * h, 1), "Fee rate %": step,
                         "Vola p.a. %": round(100 * vol), "Kurs 30T %": round(100 * (df.close.iloc[-1] / df.close.iloc[0] - 1), 1),
                         "Theorie Fee-APR %": round(100 * vol ** 2 / (4 * h)),
                         "Fee-APR %": m["Grid-Ertrag brutto $"] * k,
                         "Funding %": m["Funding $"] * k, "Gebuehren %": m["Gebuehren $"] * k,
                         "Bestand %": m["Positions-PnL $"] * k, "NETTO APR %": m["Netto-Ergebnis $"] * k,
                         "angezeigt APR %": apr_shown})
    tab = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(f"Zeitraum {start:%d.%m.} - {end:%d.%m.%Y}, Range bei jedem Austritt neu um den Preis gelegt\n")
    print(tab.to_string(index=False, float_format=lambda x: f"{x:,.1f}"))
    tab.to_csv(report_dir("liquidity") / "explore_check.csv", index=False)


if __name__ == "__main__":
    main()
