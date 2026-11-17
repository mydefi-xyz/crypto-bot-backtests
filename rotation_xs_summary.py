"""Compact comparison of saved rotation_xs runs.

    python rotation_xs_summary.py hyperliquid majors20 majors40 top40vol365
"""
import sys

import pandas as pd

from data import report_dir

pd.set_option("display.width", 250)
OUT = report_dir("rotation")
TURN = "Umsatz x Equity avg p.a."
for tag in sys.argv[1:]:
    d = pd.read_csv(OUT / f"rotation_xs_results_{tag}.csv")
    print(f"##### {tag}")
    d = d[d.Zeitraum.isin(["Vorjahr", "Letzte 12 Monate", "Gesamt 24 Monate"])]
    print(d.pivot(index="Variante", columns="Zeitraum", values=["Netto %", "Max Drawdown %"]).round(1).to_string())
    if TURN in d:  # older result files only have turnover relative to the start capital
        st = d[d.Variante == "Standard"].set_index("Zeitraum")[TURN]
        print("  Umsatz Standard (x Equity avg p.a.): "
              + ", ".join(f"{p} {v:.0f}x (0,02 % Gebuehr = {v * 0.02:.1f} % p.a.)" for p, v in st.items()))
    sweep = OUT / f"rotation_xs_sweep_{tag}.csv"
    if sweep.exists():
        s = pd.read_csv(sweep)
        for p in ("Vorjahr", "Letzte 12 Monate"):
            x = s[s.period == p]
            print(f"  Sweep {p}: {(x['Netto %'] > 0).sum()}/{len(x)} positiv, Median {x['Netto %'].median():+.1f} %")
    lag = OUT / f"rotation_xs_lag_{tag}.csv"
    if lag.exists():
        g = pd.read_csv(lag)
        g = g[g.Zeitraum.isin(["Vorjahr", "Letzte 12 Monate", "Gesamt 24 Monate"])]
        print("  Verzoegerung (Netto %):")
        print("  " + g.pivot_table(index=["Variante", "Verzoegerung Tage"], columns="Zeitraum", values="Netto %")
              .round(1).to_string().replace("\n", "\n  "))
