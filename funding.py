"""Funding backtest: Funding Rate Farming (statische Carry) vs. Funding Rate Arbitrage (dynamisch).

Farming   : dauerhaft Short-Perp auf Boerse X + Long Spot (oder Long-Perp auf Boerse Y), Funding kassieren.
Arbitrage : alle `rebalance` Stunden die Coin/Boersen-Paare mit dem hoechsten Funding-Spread waehlen
            (Short dort, wo Longs viel zahlen, Long dort, wo wenig/negativ), Positionen halten oder wechseln.

Kursrisiko ist gehedgt (gleiche Stueckzahl long und short). Modelliert sind die Margin der Perp-Beine:
taegliche Pruefung um 00:00 UTC (nur mit dem Kurs zu diesem Zeitpunkt). Bei > 25 % Kursbewegung seit dem
letzten Rebalancing oder Margin-Quote eines Perp-Beins < 15 % werden beide Beine auf den Zielhebel
zurueckgesetzt (Handelskosten auf beiden Beinen). Erreicht das Stunden-Hoch (Short) bzw. -Tief (Long-Perp)
den Liquidationskurs, ist die Margin dieses Beins verloren und das andere Bein bis zur naechsten Pruefung
ungesichert. Kontowert = Spot-Bein + Margin + offene PnL der Perp-Beine + gezahltes/erhaltenes Funding - Kosten.

Nicht modelliert: Basis (Preisunterschied Perp vs. Spot bzw. zwischen Boersen; alle Beine werden mit dem
Binance-Futures-Kurs bewertet), Kosten und Dauer von Margin-Transfers zwischen Boersen/Wallets.

    python funding_download.py     # einmalig, ~30 min (Hyperliquid drosselt)
    python funding.py
"""
from __future__ import annotations

import argparse
import itertools
import sys
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data import load_funding, load_klines, report_dir

DAY, HOUR = 86_400_000, 3_600_000
OUT = report_dir("funding")
MAJORS = ["BTC", "ETH", "SOL", "XRP", "BNB", "DOGE", "ADA", "LINK", "AVAX", "SUI", "LTC", "BCH", "TRX", "DOT",
          "TON", "XLM", "HBAR", "NEAR", "UNI", "AAVE"]
ALTS = ["HYPE", "ENA", "WIF", "1000PEPE", "FARTCOIN"]
COINS = MAJORS + ALTS
EXCHANGES = ["binance", "hyperliquid", "lighter"]
FEE = {"binance": 0.0005, "hyperliquid": 0.00045, "lighter": 0.0}   # taker, Basis-Tier
SPOT_FEE = 0.001
SLIP = 0.0003
HL_BASE_RATE = 0.0001 / 8            # Hyperliquid interest component per hour (10.95 % p.a.)
SEL_DAYS, MIN_COVER, MIN_VOL = 30, 0.9, 20e6   # coin selection: only the 30 days before the period start
MONTH_COL = "Monatsabschnitte positiv (inkl. Teilmonate)"


@dataclass(frozen=True)
class Risk:
    trigger: float = 0.25      # rebalance if price moved more than x since the last rebalance
    min_margin: float = 0.15   # ... or equity / notional of a perp leg below x
    mmr: float = 0.01          # maintenance margin (fraction of notional), liquidation when equity < mmr


def window() -> tuple[int, int]:
    end = int(datetime(2026, 11, 6, tzinfo=timezone.utc).timestamp() * 1000) // DAY * DAY
    return end - 800 * DAY, end


# --------------------------------------------------------------------------- data
def hourly_rates(coins, exchanges, s_ms, e_ms) -> dict[str, pd.DataFrame]:
    """rate charged to longs per hour (fraction). NaN = market not available on that exchange at that time."""
    idx = np.arange(s_ms, e_ms, HOUR)
    out = {}
    for ex in exchanges:
        M = pd.DataFrame(np.nan, index=idx, columns=coins)
        for c in coins:
            try:
                f = load_funding(c, ex, s_ms, e_ms)
            except Exception:  # noqa: BLE001 - not listed
                continue
            if f.empty:
                continue
            h = (f["t"].to_numpy() // HOUR) * HOUR
            s = pd.Series(f["rate"].to_numpy()).groupby(h).sum()
            col = pd.Series(0.0, index=idx)
            col[s.index[(s.index >= s_ms) & (s.index < e_ms)]] = s[(s.index >= s_ms) & (s.index < e_ms)].to_numpy()
            gap = 9 * HOUR if ex == "binance" else 2 * HOUR   # not listed before first / after last record
            col[(idx < h.min()) | (idx > h.max() + gap)] = np.nan
            M[c] = col.to_numpy()
        out[ex] = M
    return out


def hourly_prices(coins, s_ms, e_ms) -> dict[str, pd.DataFrame]:
    """Binance futures 1h candles: open (= price at the decision time), high, low, close per hour."""
    idx = np.arange(s_ms, e_ms, HOUR)
    out = {k: pd.DataFrame(np.nan, index=idx, columns=coins) for k in ("open", "high", "low", "close")}
    for c in coins:
        k = load_klines(c, "1h", s_ms, e_ms)
        if k.empty:
            continue
        k = k.set_index("t").reindex(idx)
        for f in out:
            out[f][c] = k[f].to_numpy()
    first = out["open"].notna().cummax()                     # not listed yet -> stays NaN
    out["close"] = out["close"].ffill().where(first)
    out["open"] = out["open"].fillna(out["close"].shift(1)).where(first)
    out["high"] = out["high"].fillna(out["open"])
    out["low"] = out["low"].fillna(out["open"])
    return out


def daily_volume(coins, s_ms, e_ms) -> pd.DataFrame:
    """Binance futures daily quote volume (USD), index = day start."""
    V = {}
    for c in coins:
        k = load_klines(c, "1d", s_ms, e_ms)
        if not k.empty:
            V[c] = pd.Series(k["quote_vol"].to_numpy(), index=k["t"].to_numpy())
    return pd.DataFrame(V).reindex(columns=coins)


def preselect(R: dict, V: pd.DataFrame, coins, exchanges, start_ms: int) -> list[str]:
    """Coins tradable at `start_ms`, using only the SEL_DAYS before it: funding data on every exchange for
    >= MIN_COVER of the hours and median daily Binance futures volume >= MIN_VOL."""
    a = start_ms - SEL_DAYS * DAY
    out = []
    for c in coins:
        cover = [R[e][c].loc[a:start_ms - 1].notna().mean() for e in exchanges] or [1.0]
        vol = V[c].loc[a:start_ms - 1].median() if c in V else np.nan
        if min(cover) >= MIN_COVER and vol >= MIN_VOL:
            out.append(c)
    return out


# --------------------------------------------------------------------------- hedged position
class Hedge:
    """Delta-neutral pair on one coin: short perp + long leg (spot or perp) with the same quantity.

    Funding is booked to the perp collateral. While both legs are open, price moves cancel out; if a perp leg
    is liquidated its collateral is lost and the other leg stays open (unhedged) until the next check.
    """

    def __init__(self, fee_s: float, fee_l: float, spot: bool, lev: float, mmr: float):
        self.fee_s, self.fee_l, self.spot, self.lev, self.mmr = fee_s, fee_l, spot, lev, mmr
        self.q_s = self.q_l = self.c_s = self.c_l = self.p0 = 0.0
        self.liq_ref: float | None = None   # hedged value right before a liquidation

    def value(self, px: float) -> float:
        long = self.q_l * px if self.spot else self.c_l + self.q_l * (px - self.p0)
        return long + self.c_s + self.q_s * (self.p0 - px)

    def set(self, value: float, px: float) -> float:
        """Resize both legs to the target leverage for sleeve value `value`; returns the trading cost."""
        q = self._notional(value) / px
        cost = (abs(q - self.q_s) * self.fee_s + abs(q - self.q_l) * self.fee_l) * px
        v = value - cost
        n = self._notional(v)
        self.q_s = self.q_l = n / px
        self.c_s, self.c_l = (v - n, 0.0) if self.spot else (v / 2, v / 2)
        self.p0 = px
        return cost

    def close(self, px: float) -> tuple[float, float]:
        cost = (self.q_s * self.fee_s + self.q_l * self.fee_l) * px
        v = self.value(px) - cost
        self.q_s = self.q_l = self.c_s = self.c_l = 0.0
        return v, cost

    def fund(self, px: float, r_short: float, r_long: float) -> float:
        f_s = self.q_s * px * r_short
        f_l = 0.0 if self.spot else self.q_l * px * r_long
        self.c_s += f_s
        self.c_l -= f_l
        return f_s - f_l

    def check_liq(self, hi: float, lo: float) -> str:
        """Liquidate a perp leg if this hour's high (short) or low (long perp) reaches its liquidation price."""
        hit = []
        if self.q_s > 0:
            p_liq = (self.c_s + self.q_s * self.p0) / (self.q_s * (1 + self.mmr))
            if hi >= p_liq:
                if self.liq_ref is None:
                    self.liq_ref = self.value(p_liq)
                self.q_s = self.c_s = 0.0
                hit.append("short")
        if not self.spot and self.q_l > 0:
            p_liq = (self.q_l * self.p0 - self.c_l) / (self.q_l * (1 - self.mmr))
            if lo <= p_liq:
                if self.liq_ref is None:
                    self.liq_ref = self.value(p_liq)
                self.q_l = self.c_l = 0.0
                hit.append("long")
        return "+".join(hit)

    def needs_rebalance(self, px: float, risk: Risk) -> bool:
        if self.liq_ref is not None:
            return True
        m = [(self.c_s + self.q_s * (self.p0 - px)) / (self.q_s * px)]
        if not self.spot:
            m.append((self.c_l + self.q_l * (px - self.p0)) / (self.q_l * px))
        return abs(px / self.p0 - 1) > risk.trigger or min(m) < risk.min_margin

    def _notional(self, value: float) -> float:
        return value * self.lev / (self.lev + 1) if self.spot else value * self.lev / 2


@dataclass
class Book:
    funding: float = 0.0
    costs: float = 0.0
    reb_costs: float = 0.0
    rebalances: int = 0
    liquidations: int = 0
    liq_loss: float = 0.0       # hedged value before liquidation -> value at the next check
    events: list = field(default_factory=list)

    def liq(self, t: int, coin: str, leg: str, h: Hedge):
        self.liquidations += 1
        self.events.append(dict(t=pd.to_datetime(t, unit="ms"), coin=coin, leg=leg, p0=h.p0))

    def settle_liq(self, h: Hedge, px: float):
        if h.liq_ref is not None:
            self.liq_loss += h.value(px) - h.liq_ref
            h.liq_ref = None

    def rebalance(self, h: Hedge, px: float):
        self.settle_liq(h, px)
        c = h.set(h.value(px), px)
        self.rebalances += 1
        self.costs += c
        self.reb_costs += c

    def close(self, h: Hedge, px: float) -> float:
        self.settle_liq(h, px)
        v, c = h.close(px)
        self.costs += c
        return v

    def result(self, equity: np.ndarray, **extra) -> dict:
        return dict(equity=equity, funding=self.funding, costs=-self.costs, reb_costs=-self.reb_costs,
                    rebalances=self.rebalances, liquidations=self.liquidations, liq_loss=self.liq_loss,
                    events=self.events, **extra)


# --------------------------------------------------------------------------- strategies
@dataclass(frozen=True)
class ArbCfg:
    exchanges: tuple = ("binance", "hyperliquid")
    coins: tuple = tuple(COINS)
    k: int = 5                 # max. gleichzeitige Positionen (Paare)
    rebalance_h: int = 8
    lookback_h: int = 72       # Signal = mittlerer Spread der letzten x Stunden
    min_apr: float = 0.15      # neue Position nur ab x p.a. Spread
    keep_apr: float = 0.03     # bestehende halten, solange Spread > x p.a.
    lev: float = 2.0           # Hebel je Bein (Notional / Margin)
    pool: float = 2000.0
    risk: Risk = Risk()


def _px(PX: dict, coins, i0: int, i1: int):
    return [PX[k][list(coins)].to_numpy()[i0:i1] for k in ("open", "high", "low", "close")]


def simulate_arb(R: dict, PX: dict, i0: int, i1: int, cfg: ArbCfg) -> dict:
    ex = list(cfg.exchanges)
    coins = list(cfg.coins)
    A = np.stack([R[e][coins].to_numpy() for e in ex])             # (E, T, K)
    pairs = [(a, b) for a, b in itertools.permutations(range(len(ex)), 2)]
    # spread for (short a, long b) = rate_a - rate_b   (short receives rate_a, long pays rate_b)
    S = np.stack([A[a] - A[b] for a, b in pairs])                   # (Pairs, T, K)
    csum = np.nancumsum(np.nan_to_num(S), axis=1)
    valid = np.isfinite(S)
    A0 = np.nan_to_num(A)
    O, Hi, Lo, C = _px(PX, coins, 0, A.shape[1])
    n_leg = cfg.pool * cfg.lev / (2 * cfg.k)                         # notional per leg at entry
    v_leg = 2 * n_leg / cfg.lev                                      # margin for both legs
    times = R[ex[0]].index.to_numpy()
    held: dict[int, tuple[int, Hedge]] = {}                          # coin -> (pair, position)
    book = Book()
    cash = cfg.pool
    equity = np.zeros(i1 - i0)
    held_count = np.zeros(i1 - i0)
    opens = 0
    lb = cfg.lookback_h
    for j, t in enumerate(range(i0, i1)):
        for c, (p, h) in held.items():                               # hourly funding on held positions
            a, b = pairs[p]
            book.funding += h.fund(O[t, c], A0[a, t, c], A0[b, t, c])
        if times[t] % DAY == 0:                                      # daily margin check, price at 00:00 only
            for c in list(held):
                p, h = held[c]
                if h.liq_ref is not None:                            # leg liquidated -> close the rest
                    cash += book.close(h, O[t, c])
                    del held[c]
                elif h.needs_rebalance(O[t, c], cfg.risk):
                    book.rebalance(h, O[t, c])
        if j % cfg.rebalance_h == 0 and t - lb >= 1:
            win = (csum[:, t - 1, :] - csum[:, t - lb - 1, :]) / lb   # mean spread per hour, (Pairs, K)
            ok = valid[:, t - lb:t, :].all(axis=1) & valid[:, t, :] & np.isfinite(O[t])
            sig = np.where(ok, win, -np.inf) * 24 * 365                # annualised
            best_pair = sig.argmax(axis=0)
            best_sig = sig.max(axis=0)
            keep = {c: ph for c, ph in held.items() if sig[ph[0], c] > cfg.keep_apr and ph[1].liq_ref is None}
            for c in set(held) - set(keep):
                cash += book.close(held[c][1], O[t, c])
            held = keep
            free = cfg.k - len(held)
            if free > 0:
                cand = [c for c in np.argsort(-best_sig) if c not in held and best_sig[c] > cfg.min_apr][:free]
                for c in cand:
                    p = int(best_pair[c])
                    a, b = pairs[p]
                    h = Hedge(FEE[ex[a]] + SLIP, FEE[ex[b]] + SLIP, False, cfg.lev, cfg.risk.mmr)
                    book.costs += h.set(v_leg, O[t, c])
                    cash -= v_leg
                    held[int(c)] = (p, h)
                    opens += 1
        for c, (p, h) in held.items():
            if leg := h.check_liq(Hi[t, c], Lo[t, c]):
                book.liq(times[t], coins[c], leg, h)
        equity[j] = cash + sum(h.value(C[t, c]) for c, (p, h) in held.items())
        held_count[j] = len(held)
    for c, (p, h) in held.items():                                   # close all at the end
        cash += book.close(h, C[i1 - 1, c])
    equity[-1] = cash
    return book.result(equity, opens=opens, avg_pos=float(held_count.mean()))


def simulate_farm(R: dict, PX: dict, i0: int, i1: int, ex: str, coins: list[str], pool: float, lev: float,
                  hedge: str | None = None, risk: Risk = Risk()) -> dict:
    """Static: short perp on `ex` for all coins, long spot (hedge=None) or long perp on `hedge`."""
    O, Hi, Lo, C = _px(PX, coins, i0, i1)
    rs = np.nan_to_num(R[ex][coins].to_numpy()[i0:i1])
    rl = np.nan_to_num(R[hedge][coins].to_numpy()[i0:i1]) if hedge else np.zeros_like(rs)
    fee_l = (FEE[hedge] if hedge else SPOT_FEE) + SLIP
    times = R[ex].index.to_numpy()[i0:i1]
    pos = [Hedge(FEE[ex] + SLIP, fee_l, hedge is None, lev, risk.mmr) for _ in coins]
    book = Book()
    for c, h in enumerate(pos):
        book.costs += h.set(pool / len(coins), O[0, c])
    equity = np.zeros(i1 - i0)
    for j in range(i1 - i0):
        if j and times[j] % DAY == 0:                                # daily margin check, price at 00:00 only
            for c, h in enumerate(pos):
                if h.needs_rebalance(O[j, c], risk):
                    book.rebalance(h, O[j, c])
        for c, h in enumerate(pos):
            book.funding += h.fund(O[j, c], rs[j, c], rl[j, c])
            if leg := h.check_liq(Hi[j, c], Lo[j, c]):
                book.liq(times[j], coins[c], leg, h)
        equity[j] = sum(h.value(C[j, c]) for c, h in enumerate(pos))
    equity[-1] = sum(book.close(h, C[-1, c]) for c, h in enumerate(pos))
    return book.result(equity)


def metrics(eq: np.ndarray, times: np.ndarray, pool: float) -> dict:
    days = len(eq) / 24
    s = pd.Series(eq, index=pd.to_datetime(times, unit="ms"))
    m = s.resample("ME").last()
    prev = m.shift(1)
    prev.iloc[0] = pool
    mr = 100 * (m - prev) / pool
    peak = np.maximum.accumulate(np.maximum(eq, pool))
    return {"Netto %": 100 * (eq[-1] / pool - 1), "p.a. %": 100 * (eq[-1] / pool - 1) / days * 365,
            MONTH_COL: f"{int((mr > 0).sum())}/{len(mr)}", "schlechtester Monatsabschnitt %": mr.min(),
            "bester Monatsabschnitt %": mr.max(), "Max Drawdown Kontowert %": 100 * (eq / peak - 1).min()}, mr


# --------------------------------------------------------------------------- main
def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", type=float, default=2000)
    ap.add_argument("--lev", type=float, default=2.0)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--rebalance", type=int, default=8)
    ap.add_argument("--lookback", type=int, default=72)
    ap.add_argument("--min-apr", type=float, default=15)
    ap.add_argument("--trigger", type=float, default=25, help="Rebalancing ab x %% Kursbewegung seit letztem")
    ap.add_argument("--min-margin", type=float, default=15, help="Rebalancing, wenn Margin-Quote < x %%")
    ap.add_argument("--mmr", type=float, default=1.0, help="Maintenance Margin in %% des Positionswerts")
    a = ap.parse_args()
    risk = Risk(a.trigger / 100, a.min_margin / 100, a.mmr / 100)

    s_ms, e_ms = window()
    print("Lade Funding und Kurse (Cache) ...")
    R = hourly_rates(COINS, EXCHANGES, s_ms, e_ms)
    PX = hourly_prices(COINS, s_ms, e_ms)
    V = daily_volume(COINS, s_ms, e_ms)
    times = R["binance"].index.to_numpy()
    for ex in EXCHANGES:
        print(f"  {ex:<12} Coins mit Daten: {int(R[ex].notna().any().sum())}/{len(COINS)}")

    y1, y2 = e_ms - 365 * DAY, e_ms - 730 * DAY
    per = {"Vorjahr": (y2, y1), "Letzte 12 Monate": (y1, e_ms)}
    pi = {k: (int(np.searchsorted(times, s)), int(np.searchsorted(times, e))) for k, (s, e) in per.items()}

    # ---- 1) average funding (p.a., paid by longs) per exchange and coin
    rows = []
    for p, (i0, i1) in pi.items():
        for ex in EXCHANGES:
            M = R[ex].iloc[i0:i1]
            cov = M.notna().mean()
            apr = M.mean() * 24 * 365 * 100
            neg = (M.rolling(24 * 7).sum() < 0).mean() * 100
            for c in COINS:
                if cov[c] > 0.9:
                    rows.append(dict(period=p, exchange=ex, coin=c, apr=apr[c], neg_weeks=neg[c]))
    avg = pd.DataFrame(rows)
    print("\nDurchschnittliches Funding (p.a. %, das Longs an Shorts zahlen):")
    print(avg.pivot_table(index="coin", columns=["period", "exchange"], values="apr").reindex(COINS).round(1)
          .to_string())
    print("\nBTC auf Hyperliquid: Anteil Stunden am festen Zinsanteil (0,00125 %/h), darueber, darunter, negativ:")
    for p, (i0, i1) in pi.items():
        x = R["hyperliquid"]["BTC"].iloc[i0:i1].dropna()
        at = np.isclose(x, HL_BASE_RATE, rtol=0, atol=1e-10)
        print(f"  {p:<17} am Festwert {100 * at.mean():.0f} %, darueber {100 * ((x > HL_BASE_RATE) & ~at).mean():.0f} %, "
              f"darunter {100 * ((x < HL_BASE_RATE) & ~at).mean():.0f} %, negativ {100 * (x < 0).mean():.0f} %, "
              f"Schnitt {100 * x.mean() * 24 * 365:.1f} % p.a.")

    # ---- 2) farming (static); coins chosen with data from before each period only
    res_rows, curves, months, liq_rows = [], {}, {}, []
    def add(name, p, res, coins):
        m, mr = metrics(res["equity"], times[pi[p][0]:pi[p][1]], a.pool)
        res_rows.append({"Zeitraum": p, "Strategie": name, **m,
                         "Funding USD": res["funding"], "Kosten USD": res["costs"],
                         "davon Rebalancing USD": res["reb_costs"], "Rebalancings": res["rebalances"],
                         "Liquidationen": res["liquidations"], "Liquidationsverlust USD": res["liq_loss"],
                         "Positionen avg": res.get("avg_pos", np.nan), "Wechsel": res.get("opens", np.nan),
                         "Coins": len(coins)})
        curves[(p, name)] = res["equity"]
        months[(p, name)] = mr
        liq_rows.extend(dict(Zeitraum=p, Strategie=name, **ev) for ev in res["events"])
        print(f"  {p:<17} {name:<50} {len(coins):>2} Coins: {','.join(coins)}", flush=True)

    print(f"\nCoin-Auswahl (nur Daten der {SEL_DAYS} Tage vor Beginn, Abdeckung >= {MIN_COVER:.0%}, "
          f"Binance-Volumen Median >= {MIN_VOL / 1e6:.0f} Mio USD/Tag):")
    for p, (i0, i1) in pi.items():
        for ex in ("hyperliquid", "binance"):
            for basket, cl in (("BTC+ETH", ["BTC", "ETH"]), ("20 Majors", MAJORS)):
                cl = preselect(R, V, cl, [ex], per[p][0])
                add(f"Farming: Short {ex} + Spot ({basket})", p,
                    simulate_farm(R, PX, i0, i1, ex, cl, a.pool, a.lev, risk=risk), cl)
        cl = preselect(R, V, MAJORS, ["hyperliquid", "binance"], per[p][0])
        add("Farming: Short HL + Long Binance (20 Majors)", p,
            simulate_farm(R, PX, i0, i1, "hyperliquid", cl, a.pool, a.lev, hedge="binance", risk=risk), cl)

    # ---- 3) arbitrage (dynamic)
    base = ArbCfg(k=a.k, rebalance_h=a.rebalance, lookback_h=a.lookback, min_apr=a.min_apr / 100, lev=a.lev,
                  pool=a.pool, risk=risk)

    def arb_cfg(p, coins, exchanges=base.exchanges, cfg=base):
        return replace(cfg, exchanges=tuple(exchanges), coins=tuple(preselect(R, V, coins, [], per[p][0])))

    for p, (i0, i1) in pi.items():
        for name, cl in (("Arbitrage HL<->Binance (Majors)", MAJORS), ("Arbitrage HL<->Binance (Majors+Alts)", COINS)):
            cfg = arb_cfg(p, cl)
            add(name, p, simulate_arb(R, PX, i0, i1, cfg), cfg.coins)
    p = "Letzte 12 Monate"
    lighter_ok = preselect(R, V, COINS, ["lighter"], per[p][0])
    for name, cl in (("Arbitrage 3 Boersen inkl. Lighter (Majors)", [c for c in MAJORS if c in lighter_ok]),
                     ("Arbitrage 3 Boersen inkl. Lighter (Majors+Alts)", lighter_ok)):
        cfg = arb_cfg(p, cl, EXCHANGES)
        add(name, p, simulate_arb(R, PX, *pi[p], cfg), cfg.coins)

    tab = pd.DataFrame(res_rows)
    pd.set_option("display.width", 300)
    for p in pi:
        print(f"\n=== {p} | Pool {a.pool:,.0f} USD, Hebel {a.lev}x je Bein, Rebalancing ab {a.trigger:.0f} % "
              f"Kursbewegung oder Margin-Quote < {a.min_margin:.0f} % ===")
        print(tab[tab.Zeitraum == p].drop(columns="Zeitraum").set_index("Strategie")
              .to_string(float_format=lambda x: f"{x:,.1f}"))
    print("\n'Monatsabschnitte inkl. Teilmonate': jeder Zeitraum (6.11.-5.11.) beruehrt 13 Kalendermonate, "
          "der erste und letzte sind Teilmonate.")

    # ---- 4) sweep arbitrage
    srows = []
    for p, (i0, i1) in pi.items():
        for k, rb, lb, mn in itertools.product([3, 5, 10], [1, 8, 24], [24, 72, 168], [0.10, 0.20, 0.40]):
            cfg = arb_cfg(p, COINS, cfg=replace(base, k=k, rebalance_h=rb, lookback_h=lb, min_apr=mn))
            res = simulate_arb(R, PX, i0, i1, cfg)
            m, _ = metrics(res["equity"], times[i0:i1], a.pool)
            srows.append(dict(period=p, k=k, rebalance=rb, lookback=lb, min_apr=mn, **m,
                              rebalancings=res["rebalances"], liquidations=res["liquidations"]))
        print(f"  Sweep {p} fertig", flush=True)
    sweep = pd.DataFrame(srows)
    sweep.round(2).to_csv(OUT / "funding_sweep.csv", index=False)
    tab.round(2).to_csv(OUT / "funding_results.csv", index=False)
    pd.DataFrame(liq_rows, columns=["Zeitraum", "Strategie", "t", "coin", "leg", "p0"]).to_csv(
        OUT / "funding_liquidations.csv", index=False)
    avg.round(2).to_csv(OUT / "funding_avg.csv", index=False)
    piv = sweep.pivot_table(index=["k", "rebalance", "lookback", "min_apr"], columns="period", values="p.a. %")
    print("\nSweep Arbitrage HL<->Binance (Majors+Alts), p.a. %:")
    for p in pi:
        print(f"  {p}: {(piv[p] > 0).sum()}/{len(piv)} positiv, Median {piv[p].median():+.1f} %, "
              f"min {piv[p].min():+.1f} %, max {piv[p].max():+.1f} %")
    print(f"  Korrelation zwischen den Jahren: {piv['Vorjahr'].corr(piv['Letzte 12 Monate']):+.2f}")

    # ---- report
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.08, subplot_titles=[f"Kontowert - {p}" for p in pi])
    palette = ["#00cc96", "#19d3f3", "#636efa", "#ab63fa", "#ffa15a", "#ef553b", "#fecb52", "#b6e880", "#ff97ff",
               "#aaaaaa", "#ffffff"]
    names = list(dict.fromkeys(n for _, n in curves))
    for row, p in enumerate(pi, start=1):
        ts = pd.to_datetime(times[pi[p][0]:pi[p][1]], unit="ms")
        for j, n in enumerate(names):
            if (p, n) in curves:
                fig.add_trace(go.Scattergl(x=ts[::6], y=curves[(p, n)][::6], name=n, legendgroup=n,
                                           showlegend=row == 1 or p == "Letzte 12 Monate" and "Lighter" in n,
                                           line=dict(color=palette[j % len(palette)], width=1.5)), row=row, col=1)
    fig.update_layout(height=900, template="plotly_dark", legend=dict(orientation="h", y=-0.1),
                      title=f"Funding Farming vs. Arbitrage - Pool {a.pool:,.0f} USD, Hebel {a.lev}x je Bein "
                            f"(Kontowert inkl. Margin/Liquidation der Perp-Beine, Basis nicht modelliert)")
    hm = avg.pivot_table(index="coin", columns=["period", "exchange"], values="apr").reindex(COINS)
    hfig = go.Figure(go.Heatmap(z=hm.values, y=hm.index, x=[f"{p}<br>{e}" for p, e in hm.columns], zmid=0,
                                colorscale="RdYlGn", text=np.round(hm.values, 1), texttemplate="%{text}",
                                colorbar=dict(title="% p.a.")))
    hfig.update_layout(height=700, template="plotly_dark",
                       title="Durchschnittliches Funding p.a. (positiv = Longs zahlen an Shorts)")
    css = ("<style>body{background:#111;color:#ddd;font-family:sans-serif}.t{border-collapse:collapse;margin:10px}"
           ".t td,.t th{padding:4px 10px;border-bottom:1px solid #333;text-align:right}</style>")
    note = (f"<p>Kontowert = Spot-Bein + Margin + offene PnL der Perp-Beine + Funding - Kosten. Taegliche Pruefung "
            f"um 00:00 UTC: Rebalancing beider Beine auf {a.lev}x bei &gt; {a.trigger:.0f} % Kursbewegung seit dem "
            f"letzten Rebalancing oder Margin-Quote &lt; {a.min_margin:.0f} %. Liquidation, wenn das Stunden-Hoch "
            f"(Short) bzw. -Tief (Long-Perp) den Liquidationskurs erreicht (Maintenance Margin {a.mmr:g} %): Margin "
            f"des Beins verloren, Rest bis zur naechsten Pruefung ungesichert. Nicht modelliert: Basis (alle Beine mit "
            f"Binance-Futures-Kurs bewertet), Kosten von Margin-Transfers. Coin-Auswahl nur mit Daten der "
            f"{SEL_DAYS} Tage vor Beginn. 'Monatsabschnitte inkl. Teilmonate': 13 Abschnitte je Zeitraum.</p>")
    parts = [fig.to_html(full_html=False, include_plotlyjs="cdn"), note,
             hfig.to_html(full_html=False, include_plotlyjs=False),
             "<h3>Kennzahlen</h3>", tab.round(2).to_html(index=False, border=0, classes="t", na_rep="")]
    (OUT / "funding_report.html").write_text(f"<html><head><meta charset='utf-8'>{css}</head><body>"
                                             + "".join(parts) + "</body></html>", encoding="utf-8")
    print(f"\nReport: {OUT / 'funding_report.html'}")


if __name__ == "__main__":
    main()
