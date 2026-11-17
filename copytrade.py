"""Copy-Trading-Test mit echten Hyperliquid-Konten: Sagt vergangene Performance die zukuenftige voraus?

Vorgehen (ohne Blick in die Zukunft):
  1. Zufallsstichprobe aktiver Konten aus dem oeffentlichen Leaderboard.
  2. Fuer jedes Konto: Verlauf von kumuliertem Perp-PnL und Kontostand (API "portfolio", Wochenwerte).
  3. Mehrere Zeitfenster: Auswahl-Zeitraum (z.B. 90 Tage) -> Test-Zeitraum (die 90 Tage danach).
  4. Trader nach Regeln auswaehlen, die nur den Auswahl-Zeitraum kennen, und ihre Rendite danach messen.

Rendite = zeitgewichtete Rendite (TWR): je Schritt r = PnL-Aenderung / Kontostand am vorherigen Punkt,
Schritte verkettet. Ein- und Auszahlungen verzerren sie kaum (anders als der alte ROI = PnL / Startwert,
der nur noch als Zusatzspalte mitlaeuft).

Einschluss je Beobachtung (Konto x Zeitfenster), alles zum Auswahlzeitpunkt bekannt:
  - Kontostand >= 1.000 USD zu Beginn UND am Ende des Auswahl-Zeitraums (= Beginn der Messung),
  - im Auswahl-Zeitraum aktiv (PnL veraendert sich in >= der Haelfte der Wochen).
Ein Konto, das in der Messung alles verliert, bekommt dort -100 % (TWR-Schritt auf 0).

Kopierkosten: Der Cache enthaelt kein Volumen je Zeitfenster ("portfolio" liefert nur das Gesamtvolumen).
Geschaetzt wird deshalb mit dem durchschnittlichen Umschlag ueber die ganze Kontohistorie
(Gesamtvolumen / Zeitintegral des Kontostands). Die Schaetzung dient nur der Netto-Spalte, keiner Auswahlregel.

    python copytrade.py --n 800              # Download ~15 min beim ersten Lauf (gecacht)
    python copytrade.py --n 800 --offline    # nur Cache, keine Netzwerkzugriffe
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from data import report_dir

DAY = 86_400_000
MONTH = 30 * DAY
ROOT = Path(__file__).parent
OUT = report_dir("copytrade")
CACHE = ROOT / "data" / "hl_portfolio"
CACHE.mkdir(parents=True, exist_ok=True)
LEADERBOARD = ROOT / "data" / "hl_leaderboard.json"
SESSION = requests.Session()
MIN_AV = 1000.0          # min. account value at selection start and at measurement start
MIN_STEP_AV = 100.0      # TWR steps starting below this account value are skipped (no capital at risk)


def leaderboard(refresh: bool = False, offline: bool = False) -> pd.DataFrame:
    """Leaderboard snapshot; downloaded only if missing or --refresh, never overwritten by a failed request."""
    if (refresh or not LEADERBOARD.exists()) and not offline:
        r = requests.get("https://stats-data.hyperliquid.xyz/Mainnet/leaderboard", timeout=120)
        r.raise_for_status()
        r.json()["leaderboardRows"]                  # validate before overwriting the cache
        LEADERBOARD.write_text(r.text)
    rows = json.loads(LEADERBOARD.read_text())["leaderboardRows"]
    out = []
    for r in rows:
        w = {k: v for k, v in r["windowPerformances"]}
        out.append(dict(addr=r["ethAddress"], av=float(r["accountValue"]),
                        vlm_month=float(w["month"]["vlm"]), vlm_all=float(w["allTime"]["vlm"]),
                        pnl_all=float(w["allTime"]["pnl"]), roi_all=float(w["allTime"]["roi"])))
    return pd.DataFrame(out)


def portfolio(addr: str, offline: bool = False) -> dict | None:
    f = CACHE / f"{addr}.json"
    if f.exists():
        return json.loads(f.read_text())
    if offline:
        return None
    for attempt in range(6):
        try:
            r = SESSION.post("https://api.hyperliquid.xyz/info", json={"type": "portfolio", "user": addr}, timeout=30)
            if r.status_code == 429:
                time.sleep(10 * (attempt + 1))
                continue
            r.raise_for_status()
            d = dict(r.json())["perpAllTime"]
            f.write_text(json.dumps(d))
            time.sleep(1.1)
            return d
        except requests.RequestException:
            time.sleep(3 * (attempt + 1))
    return None


class Account:
    """Weekly history of one account: account value and cumulative perp PnL."""

    def __init__(self, d: dict):
        self.t = np.array([x[0] for x in d["accountValueHistory"]], dtype=float)
        self.av = np.array([float(x[1]) for x in d["accountValueHistory"]])
        tp = np.array([x[0] for x in d["pnlHistory"]], dtype=float)
        self.pnl = np.interp(self.t, tp, [float(x[1]) for x in d["pnlHistory"]])
        self.vlm = float(d.get("vlm", 0) or 0)

    def at(self, arr, t):
        if len(self.t) == 0 or t < self.t[0] or t > self.t[-1] + 3 * DAY:
            return np.nan
        return float(np.interp(t, self.t, arr))

    def path(self, a, b):
        """Times, account values and PnL at a, all history points inside (a, b), and b."""
        inner = self.t[(self.t > a) & (self.t < b)]
        ts = np.concatenate([[a], inner, [b]])
        return ts, np.array([self.at(self.av, x) for x in ts]), np.array([self.at(self.pnl, x) for x in ts])

    def turnover(self):
        """Average monthly volume / account value over the whole history (cost estimate only, see module doc)."""
        cap = np.sum(np.maximum(self.av[:-1], 0) * np.diff(self.t)) / MONTH
        return min(self.vlm / cap, 2000) if cap > 0 else np.nan


def twr_index(av, pnl, deposit_adj: bool = True):
    """Chained time-weighted return index (starts at 1), floored at -100 %.

    Step return r = dPnL / account value at the previous point. Weekly points hide deposits made within a step
    (deposit 50k on a 1k account and lose 5k -> r = -500 %); with deposit_adj the net inflow of the step
    (dAccountValue - dPnL, if > 0) is added to the denominator, i.e. treated as available from the step start.
    """
    base = av[:-1].copy()
    if deposit_adj:
        base += np.maximum(np.diff(av) - np.diff(pnl), 0)
    r = np.zeros(len(av) - 1)
    ok = base >= MIN_STEP_AV
    r[ok] = np.maximum(np.diff(pnl)[ok] / base[ok], -1.0)
    return np.concatenate([[1.0], np.cumprod(1 + r)])


def window_stats(acc: Account, a, b, sub=3, deposit_adj: bool = True):
    """TWR, old ROI, consistency, drawdown and activity of an account between a and b (None if no data)."""
    ts, av, pnl = acc.path(a, b)
    if not (np.all(np.isfinite(av)) and np.all(np.isfinite(pnl))):
        return None
    idx = twr_index(av, pnl, deposit_adj)
    parts = np.interp([a + (b - a) * k / sub for k in range(sub + 1)], ts, idx)
    return dict(av0=av[0], av1=av[-1], pnl=pnl[-1] - pnl[0], twr=idx[-1] - 1,
                roi=(pnl[-1] - pnl[0]) / av[0] if av[0] > 0 else np.nan,
                pos_parts=int((np.diff(parts) > 0).sum()), dd=float((idx / np.maximum.accumulate(idx) - 1).min()),
                active=float((np.abs(np.diff(pnl)) > 1e-6).mean()))


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--form", type=int, default=90, help="Tage Auswahl-Zeitraum")
    ap.add_argument("--test", type=int, default=90, help="Tage Test-Zeitraum")
    ap.add_argument("--cost", type=float, default=0.075, help="Kopierkosten in %% je Volumen (Taker + Slippage)")
    ap.add_argument("--min-active", type=float, default=0.5,
                    help="Anteil der Wochen mit PnL-Aenderung im Auswahl-Zeitraum (Aktivitaet)")
    ap.add_argument("--offline", action="store_true", help="nur Cache verwenden, nichts herunterladen")
    ap.add_argument("--no-deposit-adj", action="store_true",
                    help="TWR-Schritt strikt PnL / vorheriger Kontostand (Einzahlungen in der Woche nicht beruecksichtigt)")
    ap.add_argument("--refresh", action="store_true", help="Leaderboard neu laden")
    a = ap.parse_args()

    lb = leaderboard(a.refresh, a.offline)
    active = lb[(lb.vlm_all >= 1e6)]
    print(f"Leaderboard: {len(lb):,} Konten, davon {len(active):,} mit >= 1 Mio USD Volumen (Stand heute)")
    sample = active.sample(n=min(a.n, len(active)), random_state=42)
    data, missing = {}, 0
    for k, addr in enumerate(sample.addr, start=1):
        d = portfolio(addr, a.offline)
        if d is None:
            missing += 1
        elif d.get("pnlHistory") and d.get("accountValueHistory"):
            data[addr] = Account(d)
        if k % 200 == 0:
            print(f"  Portfolios: {k}/{len(sample)}", flush=True)
    print(f"  {len(data)} Konten mit Verlauf, {missing} nicht im Cache/nicht ladbar")

    now = max(acc.t[-1] for acc in data.values())
    rows, skipped = [], dict(no_history=0, av_start=0, inactive=0, av_mid=0, test_missing=0)
    for split in range(0, 4):                       # overlapping windows, shifted by 30 days
        t_end = now - split * 30 * DAY
        t_mid, t_start = t_end - a.test * DAY, t_end - (a.test + a.form) * DAY
        for addr, acc in data.items():
            f = window_stats(acc, t_start, t_mid, deposit_adj=not a.no_deposit_adj)
            if f is None:
                skipped["no_history"] += 1
                continue
            if f["av0"] < MIN_AV:                    # all filters below use data up to t_mid only
                skipped["av_start"] += 1
                continue
            if f["active"] < a.min_active:
                skipped["inactive"] += 1
                continue
            if f["av1"] < MIN_AV:                    # nothing left to copy at measurement start
                skipped["av_mid"] += 1
                continue
            e = window_stats(acc, t_mid, t_end, sub=1, deposit_adj=not a.no_deposit_adj)
            if e is None:                            # history ends before t_end (not a total loss)
                skipped["test_missing"] += 1
                continue
            rows.append(dict(split=split, addr=addr, f_twr=f["twr"], f_roi=f["roi"], f_pnl=f["pnl"],
                             f_pos=f["pos_parts"], f_dd=f["dd"], f_av=f["av0"], e_av=e["av0"],
                             e_twr=e["twr"], e_roi=e["roi"], e_active=e["active"],
                             cost=acc.turnover() * a.test / 30 * a.cost / 100))
    R = pd.DataFrame(rows)
    R["e_twr"] = R.e_twr.clip(-1, 5)                 # -100 % = total loss; cap outliers at +500 %
    R["e_net"] = (R.e_twr - R.cost.fillna(0)).clip(lower=-1)
    n_cand = len(R) + sum(skipped.values())
    print(f"  {n_cand:,} Kandidaten (Konto x Zeitfenster), ausgeschlossen: "
          + ", ".join(f"{k} {v:,}" for k, v in skipped.items()))
    print(f"  {len(R):,} Beobachtungen, {R.addr.nunique()} verschiedene Konten "
          f"(Zeitfenster ueberlappen -> Beobachtungen nicht unabhaengig)\n")

    def grp(name, sel):
        x = R[sel]
        return dict(Regel=name, Beobachtungen=len(x), Konten=x.addr.nunique(), **{
            "Auswahl TWR Median %": 100 * x.f_twr.median(),
            "Danach TWR Median %": 100 * x.e_twr.median(),
            "Danach TWR Mittel %": 100 * x.e_twr.mean(),
            "Danach im Plus %": 100 * (x.e_twr > 0).mean(),
            "Danach -50% oder schlechter %": 100 * (x.e_twr <= -0.5).mean(),
            "Danach Totalverlust (<= -99%) %": 100 * (x.e_twr <= -0.99).mean(),
            "Danach ohne Handel %": 100 * (x.e_active == 0).mean(),
            "Danach ROI alt Median %": 100 * x.e_roi.median(),
            "Kopierkosten Median % (Schaetzung)": 100 * x.cost.median(),
            "Netto nach Kopierkosten Mittel %": 100 * x.e_net.mean(),
        })

    by = R.groupby("split")
    top10 = R.f_twr >= by.f_twr.transform(lambda s: s.quantile(0.9))
    bot10 = R.f_twr <= by.f_twr.transform(lambda s: s.quantile(0.1))
    top_pnl = R.f_pnl >= by.f_pnl.transform(lambda s: s.quantile(0.9))
    steady = (R.f_pos == 3) & (R.f_dd > -0.15) & (R.f_twr > 0.05)
    big_steady = steady & (R.f_av >= 100_000)
    tab = pd.DataFrame([
        grp("Alle Konten (Basis)", R.f_twr.notna()),
        grp("Top 10 % nach TWR", top10),
        grp("Top 10 % nach USD-Gewinn", top_pnl),
        grp("Flop 10 % nach TWR", bot10),
        grp("Konstant: 3/3 Monate Plus, DD < 15 %, TWR > 5 %", steady),
        grp("  ... und Konto >= 100k USD", big_steady),
    ])
    pd.set_option("display.width", 300)
    print(f"Auswahl {a.form} Tage -> Test {a.test} Tage, 4 ueberlappende Zeitfenster, Rendite = TWR "
          f"({'ohne' if a.no_deposit_adj else 'mit'} Einzahlungs-Korrektur), "
          f"Kopierkosten {a.cost} % je Volumen (Umschlag geschaetzt aus Gesamtvolumen / Kontowert-Historie)")
    print(tab.to_string(index=False, float_format=lambda x: f"{x:,.1f}"))
    rho = by.apply(lambda g: g.f_twr.rank().corr(g.e_twr.rank()), include_groups=False)
    rho_old = by.apply(lambda g: g.f_roi.rank().corr(g.e_roi.rank()), include_groups=False)
    print(f"\nRang-Korrelation Auswahl-TWR vs. Test-TWR je Zeitfenster: {', '.join(f'{x:+.2f}' for x in rho)}")
    print(f"  (alter ROI, gleiche Beobachtungen: {', '.join(f'{x:+.2f}' for x in rho_old)})")
    dec = R.assign(dec=by.f_twr.transform(lambda s: pd.qcut(s.rank(method='first'), 10, labels=False)))
    print("\nTest-TWR (Median %) je Zehntel des Auswahl-TWR (0 = schlechteste, 9 = beste):")
    print((dec.groupby("dec").e_twr.median() * 100).round(1).to_string())
    print(f"\nKonten in allen 4 Zeitfenstern: {(R.groupby('addr').split.nunique() == 4).sum()}")
    tag = "_no_deposit_adj" if a.no_deposit_adj else ""
    tab.round(2).to_csv(OUT / f"copytrade_results{tag}.csv", index=False)
    R.to_csv(OUT / f"copytrade_obs{tag}.csv", index=False)
    pd.DataFrame([dict(Grund=k, Anzahl=v) for k, v in skipped.items()]).to_csv(OUT / f"copytrade_excluded{tag}.csv",
                                                                               index=False)


if __name__ == "__main__":
    main()
