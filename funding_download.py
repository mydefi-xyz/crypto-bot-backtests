"""Download funding history (Hyperliquid, Lighter, Binance) and Binance futures candles (1h for margin and
liquidation checks, 1d for the volume-based coin selection) for the funding backtest. Cached in data/.

    python funding_download.py
"""
from __future__ import annotations

import sys

from data import load_funding, load_klines
from funding import COINS, EXCHANGES, window

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    s_ms, e_ms = window()
    for ex in EXCHANGES:
        for k, c in enumerate(COINS, start=1):
            try:
                f = load_funding(c, ex, s_ms, e_ms)
                print(f"{ex:<12}{k:>3}/{len(COINS)} {c:<10} {len(f):>6} Eintraege", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"{ex:<12}{k:>3}/{len(COINS)} {c:<10} FEHLER {e}", flush=True)
    for iv in ("1h", "1d"):
        for k, c in enumerate(COINS, start=1):
            try:
                print(f"klines {iv:<5}{k:>3}/{len(COINS)} {c:<10} {len(load_klines(c, iv, s_ms, e_ms)):>6} Kerzen",
                      flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"klines {iv:<5}{k:>3}/{len(COINS)} {c:<10} FEHLER {e}", flush=True)
