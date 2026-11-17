"""Backtest verbreiteter Signal-Setups, wie sie Signal-Bots anbieten.

Setups: ATH rejection (4h), EMA pullback (4h), MACD + RSI (4h), Volume profile (4h), Breakout (1d).
Stop/Take-Profit wie in den Setup-Beschreibungen (TP 2R bzw. POC). Pro Coin und Setup max. eine Position gleichzeitig.
Kosten: 0,045 % Taker + 0,03 % Slippage je Seite, dazu Binance-Funding waehrend der Haltedauer.
Stop-vor-TP, wenn beides in derselben Kerze liegt. Limit-Einstieg (EMA pullback): Die Fill-Kerze wird mit 1h-Kerzen
aufgeloest. In der 1h-Kerze des Fills zaehlt der Stop, sobald er erreichbar ist, das Ziel nur, wenn es sicher nach dem
Fill erreicht wurde (sonst laeuft der Trade weiter).

Vergleich: exakt dieselben Trades (Einstieg, Zeitpunkt, Stop/TP-Abstand), aber zufaellige Richtung, gespiegelt um den
tatsaechlichen Einstieg (50 Durchlaeufe, eigener Zufallsstrom je Setup und Coin). Kennzahl: Anzahl Durchlaeufe mit
mindestens so gutem Ø R und empirischer p-Wert. Ist ein Setup nicht besser als Zufall, hat es keinen Richtungsvorteil.
Signifikanz: t-Wert ueber Wochensummen (viele Trades laufen gleichzeitig auf korrelierten Coins), naiver t-Wert nur
zum Vergleich.

Coin-Liste = 21 heute grosse Coins (Survivorship: abgestuerzte oder delistete Coins fehlen, Ergebnis eher zu freundlich).

    python signals.py
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from data import load_funding, load_klines, report_dir

DAY = 86_400_000
WEEK = 7 * DAY
OUT = report_dir("signals")
N_RANDOM = 50                                                           # random-direction runs
SEED = 1                                                                # base seed, stream per (setup, coin)
FUNDING = "binance"
# Survivorship: coins chosen by today's size; coins that crashed or were delisted in the period are missing.
COINS = ["BTC", "ETH", "SOL", "HYPE", "XRP", "BNB", "DOGE", "ADA", "LINK", "AVAX", "SUI", "LTC", "BCH", "TRX",
         "DOT", "TON", "XLM", "HBAR", "NEAR", "UNI", "AAVE"]
PICKS = {  # typischerweise je Setup empfohlene Coins
    "EMA pullback": ["ETH", "HYPE", "SOL"], "MACD + RSI": ["BTC", "ETH"], "Volume profile": ["BTC", "SOL", "HYPE"],
    "ATH rejection": ["ETH", "HYPE", "SOL"], "Breakout": ["BTC", "ETH", "SOL", "HYPE"]}
COST = 2 * (0.00045 + 0.0003)


# --------------------------------------------------------------------------- indicators
def ema(x, n):
    return pd.Series(x).ewm(span=n, adjust=False).mean().to_numpy()


def wilder(x, n):
    return pd.Series(x).ewm(alpha=1 / n, adjust=False).mean().to_numpy()


def indicators(df: pd.DataFrame) -> dict:
    o, h, l, c, v = (df[k].to_numpy() for k in ("open", "high", "low", "close", "quote_vol"))
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    atr = wilder(tr, 14)
    d = np.diff(c, prepend=c[0])
    rsi = 100 - 100 / (1 + wilder(np.clip(d, 0, None), 14) / np.maximum(wilder(np.clip(-d, 0, None), 14), 1e-12))
    macd = ema(c, 12) - ema(c, 26)
    sig = ema(macd, 9)
    upm, dnm = h - np.r_[h[0], h[:-1]], np.r_[l[0], l[:-1]] - l
    pdi = 100 * wilder(np.where((upm > dnm) & (upm > 0), upm, 0.0), 14) / np.maximum(atr, 1e-12)
    ndi = 100 * wilder(np.where((dnm > upm) & (dnm > 0), dnm, 0.0), 14) / np.maximum(atr, 1e-12)
    adx = wilder(100 * np.abs(pdi - ndi) / np.maximum(pdi + ndi, 1e-12), 14)
    return dict(t=df["t"].to_numpy(), o=o, h=h, l=l, c=c, v=v, atr=atr, rsi=rsi, macd=macd, sig=sig,
                hist=macd - sig, adx=adx, e50=ema(c, 50), e100=ema(c, 100), e200=ema(c, 200), n=len(c),
                step=int(np.median(np.diff(df["t"].to_numpy()))))


# --------------------------------------------------------------------------- setups
# each returns list of (i, side, stop, tp, limit, valid_bars); tp None = 2R from entry, limit None = market at o[i+1]
def s_ath(X):
    h, l, o, c, atr, rsi, e50 = X["h"], X["l"], X["o"], X["c"], X["atr"], X["rsi"], X["e50"]
    hh = pd.Series(h).rolling(180).max().to_numpy()                    # 30-Tage-Hoch (4h)
    out, k, peak = [], -10**9, 0.0
    for i in range(200, X["n"] - 1):
        if h[i] >= hh[i]:
            if rsi[i] > 70 or h[i] > e50[i] + 2.5 * atr[i]:
                k, peak = i, h[i]
            else:
                k = -10**9
        if i - k > 42:                                                  # Hoch aelter als 7 Tage
            continue
        peak = max(peak, h[i])
        rng, wick = h[i] - l[i], h[i] - max(o[i], c[i])
        if rng > 0 and wick >= 0.5 * rng and c[i] < o[i] and h[i] >= peak - 0.5 * atr[i]:
            out.append((i, -1, peak + 0.2 * atr[i], None, None, 0))
    return out


def s_ema(X):
    c, atr, hist, e100, e200 = X["c"], X["atr"], X["hist"], X["e100"], X["e200"]
    short = (c < e200) & (c < e100) & (e100 - c < 1.5 * atr) & (hist < 0)
    long_ = (c > e200) & (c > e100) & (c - e100 < 1.5 * atr) & (hist > 0)
    out = []
    for i in range(201, X["n"] - 1):
        if short[i] and not short[i - 1]:
            lim = e100[i] - 0.3 * atr[i]
            stop = e100[i] + 0.9 * atr[i]
            out.append((i, -1, stop, lim - 2 * (stop - lim), lim, 6))
        elif long_[i] and not long_[i - 1]:
            lim = e100[i] + 0.3 * atr[i]
            stop = e100[i] - 0.9 * atr[i]
            out.append((i, 1, stop, lim + 2 * (lim - stop), lim, 6))
    return out


def s_macd(X):
    h, l, o, c, atr, rsi, macd, sig, hist, adx = (X[k] for k in ("h", "l", "o", "c", "atr", "rsi", "macd", "sig",
                                                                  "hist", "adx"))
    dn = (macd < sig) & (np.r_[False, macd[:-1] >= sig[:-1]])
    up = (macd > sig) & (np.r_[False, macd[:-1] <= sig[:-1]])
    out, used = [], set()
    for i in range(50, X["n"] - 1):
        for side, cross, extreme in ((-1, dn, lambda r: r.max() >= 65), (1, up, lambda r: r.min() <= 35)):
            ks = np.flatnonzero(cross[max(0, i - 5):i + 1])
            if not len(ks):
                continue
            k = max(0, i - 5) + ks[-1]
            if (k, side) in used or not extreme(rsi[max(0, k - 12):k + 1]) or adx[i] <= 20:
                continue
            if side < 0 and hist[i] < 0 and c[i] < o[i]:
                out.append((i, -1, h[max(0, i - 20):i + 1].max() + 0.1 * atr[i], None, None, 0))
                used.add((k, side))
            elif side > 0 and hist[i] > 0 and c[i] > o[i]:
                out.append((i, 1, l[max(0, i - 20):i + 1].min() - 0.1 * atr[i], None, None, 0))
                used.add((k, side))
    return out


def profile(h, l, c, v, bins=50):
    lo, hi = l.min(), h.max()
    tp = (h + l + c) / 3
    vol, edges = np.histogram(tp, bins=bins, range=(lo, hi), weights=v)
    p = int(vol.argmax())
    a = b = p
    tot, acc = vol.sum(), vol[p]
    while acc < 0.7 * tot and (a > 0 or b < bins - 1):
        left = vol[a - 1] if a > 0 else -1
        right = vol[b + 1] if b < bins - 1 else -1
        if right >= left:
            b += 1
            acc += vol[b]
        else:
            a -= 1
            acc += vol[a]
    mid = (edges[:-1] + edges[1:]) / 2
    return mid[p], edges[b + 1], edges[a], vol[p] >= 2 * np.median(vol[vol > 0])


def s_vp(X):
    h, l, o, c, v, atr = X["h"], X["l"], X["o"], X["c"], X["v"], X["atr"]
    out, W = [], 126                                                     # 21 Tage auf 4h
    for i in range(W, X["n"] - 1):
        poc, vah, val, cluster = profile(h[i - W:i], l[i - W:i], c[i - W:i], v[i - W:i])
        if not cluster:
            continue
        rng = h[i] - l[i]
        if rng <= 0:
            continue
        a = atr[i]
        if l[i] <= val + 0.3 * a and c[i] > val and c[i] > o[i] and (min(o[i], c[i]) - l[i]) >= 0.4 * rng:
            stop = min(val, l[i]) - 0.5 * a
            if (poc - c[i]) >= 1.5 * (c[i] - stop):
                out.append((i, 1, stop, poc, None, 0))
        elif h[i] >= vah - 0.3 * a and c[i] < vah and c[i] < o[i] and (h[i] - max(o[i], c[i])) >= 0.4 * rng:
            stop = max(vah, h[i]) + 0.5 * a
            if (c[i] - poc) >= 1.5 * (stop - c[i]):
                out.append((i, -1, stop, poc, None, 0))
    return out


def s_breakout(X):
    h, l, c, v, atr, hist, e50 = X["h"], X["l"], X["c"], X["v"], X["atr"], X["hist"], X["e50"]
    hi40 = pd.Series(h).rolling(40).max().shift(1).to_numpy()
    lo40 = pd.Series(l).rolling(40).min().shift(1).to_numpy()
    vma = pd.Series(v).rolling(20).mean().shift(1).to_numpy()
    out, pending = [], []                                                # (side, level, until)
    for i in range(60, X["n"] - 1):
        if c[i] > hi40[i] and v[i] > 1.5 * vma[i] and c[i] > e50[i] and hist[i] > 0:
            pending.append((1, hi40[i], i + 10))
            continue
        if c[i] < lo40[i] and v[i] > 1.5 * vma[i] and c[i] < e50[i] and hist[i] < 0:
            pending.append((-1, lo40[i], i + 10))
            continue
        keep = []
        for side, lvl, until in pending:
            if i > until:
                continue
            a = atr[i]
            if side > 0 and l[i] <= lvl + 0.25 * a and c[i] > lvl:
                out.append((i, 1, lvl - 1.0 * a, None, None, 0))
            elif side < 0 and h[i] >= lvl - 0.25 * a and c[i] < lvl:
                out.append((i, -1, lvl + 1.0 * a, None, None, 0))
            else:
                keep.append((side, lvl, until))
        pending = keep
    return out


SETUPS = {"ATH rejection": (s_ath, "4h", 60), "EMA pullback": (s_ema, "4h", 60), "MACD + RSI": (s_macd, "4h", 120),
          "Volume profile": (s_vp, "4h", 60), "Breakout": (s_breakout, "1d", 30)}


# --------------------------------------------------------------------------- trade simulation
def fill(X, sig):
    """(fill index, entry price, filled intrabar?) or None if the limit order is not filled in time."""
    i, side, _, _, lim, valid = sig
    o, h, l = X["o"], X["h"], X["l"]
    if lim is None:
        return i + 1, o[i + 1], False
    for j in range(i + 1, min(X["n"], i + 1 + valid)):
        if (side < 0 and h[j] >= lim) or (side > 0 and l[j] <= lim):
            e = max(lim, o[j]) if side < 0 else min(lim, o[j])
            return j, e, e != o[j]
    return None


def after_fill(o, e, c, level):
    """Intrabar limit fill: price went from the open to e first. True only if `level` was surely touched after the
    fill, i.e. it lies beyond e (away from the open) or the candle closed at/beyond it."""
    d = np.sign(o - e)
    return (level - e) * d < 0 or (c - level) * d >= 0


def candle_exit(o, h, l, c, side, stop, tp, e=None):
    """(exit price, reason) within one candle or None. Stop first if both are reachable.
    e given: the limit order was filled at e inside this candle, so the target only counts if surely after the fill."""
    hit_s = l <= stop if side > 0 else h >= stop
    hit_t = h >= tp if side > 0 else l <= tp
    if e is not None:
        if hit_s:
            return stop, "stop"
        return (tp, "target") if hit_t and after_fill(o, e, c, tp) else None
    if hit_s:
        return (min(stop, o) if side > 0 else max(stop, o)), "stop"
    if hit_t:
        return (max(tp, o) if side > 0 else min(tp, o)), "target"
    return None


def fill_candle_exit(X, j0, e, side, stop, tp):
    """Exit inside the candle of an intrabar limit fill. 1h candles (if loaded) locate the fill; only the 1h candle
    containing the fill keeps the unknown-order rule. Without 1h data the rule applies to the whole candle."""
    o = X["o"][j0]
    if X.get("sub") is not None:
        st, so, sh, sl, sc = X["sub"]
        a, b = np.searchsorted(st, [X["t"][j0], X["t"][j0] + X["step"]])
        d = np.sign(o - e)
        for s in range(a, b):
            if (sl[s] <= e) if d > 0 else (sh[s] >= e):                 # first 1h candle that reaches the limit
                at_open = (so[s] - e) * d <= 0                           # opened at/beyond it -> all after the fill
                x = candle_exit(so[s], sh[s], sl[s], sc[s], side, stop, tp, None if at_open else e)
                for s2 in range(s + 1, b):
                    if x is not None:
                        break
                    x = candle_exit(so[s2], sh[s2], sl[s2], sc[s2], side, stop, tp)
                return x
    return candle_exit(o, X["h"][j0], X["l"][j0], X["c"][j0], side, stop, tp, e)


def exit_bar(X, j0, e, side, stop, tp, intrabar, max_bars):
    """(exit index, exit price, reason)."""
    o, h, l, c = X["o"], X["h"], X["l"], X["c"]
    jx = min(X["n"] - 1, j0 + max_bars)
    if intrabar:
        x = fill_candle_exit(X, j0, e, side, stop, tp)
        if x is not None:
            return j0, *x
    for j in range(j0 + intrabar, jx + 1):
        x = candle_exit(o[j], h[j], l[j], c[j], side, stop, tp)
        if x is not None:
            return j, *x
    return jx, c[jx], "time"


def funding_sum(F, t0, t1):
    """Sum of funding rates settled in (t0, t1]; F = (times, cumulative rates with leading 0)."""
    if F is None:
        return 0.0
    ft, cum = F
    return cum[np.searchsorted(ft, t1, "right")] - cum[np.searchsorted(ft, t0, "right")]


def run(X, sigs, max_bars, F=None, rng=None):
    """Simulate signals. Without rng: the setup (one position per coin). With rng: exactly the given signals
    (pass the setup's taken ones), each with random direction mirrored around the actual entry."""
    t, atr, step = X["t"], X["atr"], X["step"]
    trades, busy = [], -1
    for k, sig in enumerate(sigs):
        i, side, stop, tp, lim, valid = sig
        if rng is None and i <= busy:
            continue
        f = fill(X, sig)
        if f is None:
            busy = i + valid
            continue
        j0, e, intrabar = f
        risk = (e - stop) * side
        if risk <= 0.2 * atr[i]:
            continue
        if tp is None:
            tp = e + side * 2 * risk
        if (tp - e) * side <= 0:
            continue
        if rng is not None and rng.random() < 0.5:                       # flip: same entry, same distances
            side, stop, tp = -side, 2 * e - stop, 2 * e - tp
        jx, px, why = exit_bar(X, j0, e, side, stop, tp, intrabar, max_bars)
        t_in = t[j0] + (step // 2 if intrabar else 0)                    # intrabar times approximated by mid-candle
        t_out = t[jx] + (step if why == "time" else step // 2)
        fund = side * funding_sum(F, t_in, t_out)                        # long pays positive funding
        ret = side * (px / e - 1) - COST - fund
        trades.append(dict(k=k, t=int(t[j0]), t_exit=int(t[jx] + step), side=side, R=ret / (risk / e),
                           fund_R=-fund / (risk / e), ret=ret, bars=jx - j0, exit=why))
        busy = jx
    return trades


def equity(tr: pd.DataFrame, risk=0.01):
    """Compounded return in % if every trade risks `risk` of the realised balance at entry; max. open trades."""
    R = tr["R"].to_numpy()
    ev = sorted([(x, 0, k) for k, x in enumerate(tr["t_exit"])] + [(x, 1, k) for k, x in enumerate(tr["t"])])
    eq, stake, peak = 1.0, {}, 0
    for _, is_entry, k in ev:                                            # exits before entries at equal times
        if is_entry:
            stake[k] = risk * eq
            peak = max(peak, len(stake))
        else:
            eq += stake.pop(k) * R[k]
    return 100 * (eq - 1), peak


def stats(tr: pd.DataFrame, t0: int, t1: int) -> dict:
    if tr.empty:
        return {"Trades": 0}
    r = tr["R"]
    n_w = -(-(t1 - t0) // WEEK)
    wk = r.groupby((tr["t"] - t0) // WEEK).sum().reindex(range(n_w), fill_value=0.0)  # incl. weeks without trades
    comp, peak = equity(tr)
    win = r[r > 0]
    return {"Trades": len(r), "Trefferquote %": 100 * (r > 0).mean(), "Ø R netto": r.mean(), "Summe R": r.sum(),
            "t-Wert (Wochen)": wk.mean() / (wk.std() / np.sqrt(n_w)),
            "Rendite % einfach (1 % Risiko)": r.sum(), "Rendite % Zinseszins (1 % Risiko)": comp,
            "t-Wert naiv": r.mean() / (r.std() / np.sqrt(len(r))) if len(r) > 1 else np.nan,
            "Ø Gewinner R": win.mean() if len(win) else np.nan, "Funding Ø R": tr["fund_R"].mean(),
            "Zeitablauf %": 100 * (tr["exit"] == "time").mean(),
            "Summe R Long": r[tr["side"] > 0].sum(), "Summe R Short": r[tr["side"] < 0].sum(),
            "Ø Haltedauer (Kerzen)": tr["bars"].mean(), "max. gleichzeitig offen": peak}


# --------------------------------------------------------------------------- main
def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    end = int(datetime(2026, 11, 6, tzinfo=timezone.utc).timestamp() * 1000) // DAY * DAY
    start = end - 730 * DAY
    load_from = start - 120 * DAY                                         # Vorlauf fuer EMA 200 / Profile
    per = {"Vorjahr": (start, end - 365 * DAY), "Letzte 12 Monate": (end - 365 * DAY, end), "Gesamt": (start, end)}

    data = {}
    for iv in ("4h", "1d"):
        for c in COINS:
            df = load_klines(c, iv, load_from, end)
            if len(df) > 250:
                data[(c, iv)] = indicators(df)
    print(f"Kerzen geladen: {len(data)} Coin/Intervall-Kombinationen")
    for c in COINS:                                                      # 1h candles locate intrabar limit fills
        X = data.get((c, "4h"))
        if X is not None:
            s = load_klines(c, "1h", end - 800 * DAY, end)
            X["sub"] = tuple(s[k].to_numpy() for k in ("t", "open", "high", "low", "close")) if len(s) else None

    fund = {}
    for c in COINS:
        try:
            f = load_funding(c, FUNDING, end - 800 * DAY, end)           # same range as funding.py -> cache hit
        except Exception as ex:                                          # offline and not cached
            print(f"  Funding {c}: nicht verfuegbar ({ex})")
            continue
        if len(f):
            f = f.sort_values("t")
            fund[c] = (f["t"].to_numpy(np.int64), np.r_[0.0, np.cumsum(f["rate"].to_numpy(float))])
    no_fund = [c for c in COINS if c not in fund]
    print(f"Funding ({FUNDING}): {len(fund)} Coins" + (f", ohne: {', '.join(no_fund)}" if no_fund else ""))

    all_tr, base_rows = [], []
    for si, (name, (fn, iv, mb)) in enumerate(SETUPS.items()):
        for ci, c in enumerate(COINS):
            X = data.get((c, iv))
            if X is None:
                continue
            sigs = [s for s in fn(X) if X["t"][s[0]] >= start]
            tr = run(X, sigs, mb, fund.get(c))
            all_tr += [dict(setup=name, coin=c, **t) for t in tr]
            taken = [sigs[t["k"]] for t in tr]
            rng = np.random.default_rng([SEED, si, ci])                  # independent, reproducible per setup/coin
            for b in range(N_RANDOM):
                base_rows += [dict(setup=name, coin=c, run=b, **t) for t in run(X, taken, mb, fund.get(c), rng)]
        print(f"  {name}: fertig")
    T = pd.DataFrame(all_tr).drop(columns="k")
    B = pd.DataFrame(base_rows)
    T["period"] = np.where(T.t < per["Letzte 12 Monate"][0], "Vorjahr", "Letzte 12 Monate")
    B["period"] = np.where(B.t < per["Letzte 12 Monate"][0], "Vorjahr", "Letzte 12 Monate")

    rows, runs = [], []
    for name in SETUPS:
        for scope, coins in (("21 Coins", COINS), ("Auswahl-Coins", PICKS[name])):
            for p, (t0, t1) in per.items():
                sel = (T.setup == name) & T.coin.isin(coins) & ((T.period == p) | (p == "Gesamt"))
                bsel = (B.setup == name) & B.coin.isin(coins) & ((B.period == p) | (p == "Gesamt"))
                s = stats(T[sel], t0, t1)
                rr = B[bsel].groupby("run")["R"].agg(["size", "mean", "sum"]).reindex(range(N_RANDOM))
                rr["size"] = rr["size"].fillna(0).astype(int)
                ge = int((rr["mean"] >= s["Ø R netto"]).sum()) if s["Trades"] else np.nan
                s = {**{k: s[k] for k in list(s)[:5]}, "Zufall Ø R": rr["mean"].mean(),
                     "Zufallsläufe ≥ Setup": ge, "p empirisch": (1 + ge) / (1 + N_RANDOM),
                     **{k: s[k] for k in list(s)[5:]}}
                rows.append(dict(Setup=name, Coins=scope, Zeitraum=p, **s))
                runs += [dict(Setup=name, Coins=scope, Zeitraum=p, Lauf=b, Trades=int(x["size"]),
                              **{"Ø R": x["mean"], "Summe R": x["sum"]}) for b, x in rr.iterrows()]
    tab = pd.DataFrame(rows)
    RUNS = pd.DataFrame(runs)
    main_cols = ["Setup", "Zeitraum", "Trades", "Trefferquote %", "Ø R netto", "Summe R", "t-Wert (Wochen)",
                 "Zufall Ø R", "Zufallsläufe ≥ Setup", "p empirisch", "Rendite % Zinseszins (1 % Risiko)"]
    pd.set_option("display.width", 250)
    for scope in ("21 Coins", "Auswahl-Coins"):
        print(f"\n=== {scope} | Kosten {COST * 100:.2f} % pro Trade (hin+rueck) + Funding ===")
        print(tab[tab.Coins == scope][main_cols].set_index(["Setup", "Zeitraum"])
              .to_string(float_format=lambda x: f"{x:,.3f}"))
    tab.round(3).to_csv(OUT / "signals_results.csv", index=False)
    RUNS.round(4).to_csv(OUT / "signals_random_runs.csv", index=False)
    T.to_csv(OUT / "signals_trades.csv", index=False)

    # report: cumulative R per setup (21 coins)
    fig = go.Figure()
    for name in SETUPS:
        d = T[T.setup == name].sort_values("t")
        fig.add_trace(go.Scatter(x=pd.to_datetime(d.t, unit="ms"), y=d.R.cumsum(), name=f"{name} ({len(d)} Trades)",
                                 mode="lines"))
    fig.add_vline(x=pd.to_datetime(per["Letzte 12 Monate"][0], unit="ms"), line_dash="dot", line_color="#888")
    fig.update_layout(height=550, template="plotly_dark", yaxis_title="kumulierte R (netto)",
                      title="Signal-Setups - kumuliertes Ergebnis in R (1 R = Risiko pro Trade), 21 Coins, "
                            "nach Kosten und Funding")

    # report: distribution of the random runs vs. setup (21 coins, whole period)
    g = tab[(tab.Coins == "21 Coins") & (tab.Zeitraum == "Gesamt")].set_index("Setup")
    fig2 = go.Figure()
    for name in SETUPS:
        d = RUNS[(RUNS.Setup == name) & (RUNS.Coins == "21 Coins") & (RUNS.Zeitraum == "Gesamt")]
        fig2.add_trace(go.Box(y=d["Summe R"], name=name, boxpoints="all", jitter=0.4, pointpos=0,
                              marker=dict(size=4, color="#8ab"), line=dict(color="#8ab"), showlegend=False))
    fig2.add_trace(go.Scatter(x=list(SETUPS), y=g.loc[list(SETUPS), "Summe R"], mode="markers", name="Setup",
                              marker=dict(symbol="diamond", size=14, color="#f60")))
    fig2.update_layout(height=450, template="plotly_dark", yaxis_title="Summe R (24 Monate)",
                       title=f"Setup (Raute) vs. {N_RANDOM} Zufallsläufe (Punkte): dieselben Trades, zufällige "
                             "Richtung, 21 Coins")
    q = (RUNS.groupby(["Setup", "Coins", "Zeitraum"], sort=False)["Summe R"]
         .describe(percentiles=[0.05, 0.5, 0.95])[["min", "5%", "50%", "95%", "max"]].add_prefix("Zufall Summe R ")
         .reset_index())
    q = tab[["Setup", "Coins", "Zeitraum", "Trades", "Summe R", "Zufallsläufe ≥ Setup", "p empirisch"]].merge(q)

    css = ("<style>body{background:#111;color:#ddd;font-family:sans-serif;margin:16px}.t{border-collapse:collapse;"
           "margin:10px}.t td,.t th{padding:4px 10px;border-bottom:1px solid #333;text-align:right}"
           "li{margin:3px 0}</style>")
    notes = [
        f"Zeitraum {pd.to_datetime(start, unit='ms'):%Y-%m-%d} bis {pd.to_datetime(end, unit='ms'):%Y-%m-%d} "
        f"(Vorjahr / letzte 12 Monate), Binance-USDT-M-Kerzen.",
        f"Kosten {COST * 100:.2f} % pro Trade (Taker 0,045 % + Slippage 0,03 % je Seite) plus {FUNDING.title()}-Funding "
        "während der Haltedauer" + (f" (ohne Funding-Daten: {', '.join(no_fund)})" if no_fund else "") + ".",
        "Stop vor Ziel, wenn beides in derselben Kerze erreichbar ist. Limit-Einstieg (EMA pullback): Die Fill-Kerze "
        "wird mit 1h-Kerzen aufgelöst. In der 1h-Kerze des Fills zählt der Stop, sobald er erreichbar ist, das Ziel "
        "nur, wenn es sicher nach dem Fill erreicht wurde.",
        f"Zufalls-Vergleich: exakt dieselben Trades, Richtung per Münzwurf und um den tatsächlichen Einstieg "
        f"gespiegelt, {N_RANDOM} Läufe, eigener Zufallsstrom je Setup und Coin. p empirisch = "
        f"(1 + Läufe mit Ø R ≥ Setup) / {N_RANDOM + 1}.",
        "Haupt-Signifikanz: t-Wert über Wochensummen (inkl. Wochen ohne Trade), weil viele Trades gleichzeitig auf "
        "korrelierten Coins laufen. Der naive t-Wert je Trade ist nur zum Vergleich angegeben.",
        "Zinseszins: jeder Trade riskiert 1 % des realisierten Kontostands beim Einstieg (überlappende Trades erlaubt).",
        "Survivorship: Die 21 Coins sind nach heutiger Größe gewählt. Coins, die im Zeitraum abgestürzt oder "
        "delistet wurden, fehlen. Das macht das Ergebnis eher zu freundlich.",
    ]
    def tbl(d):
        return d.round(3).to_html(index=False, border=0, classes="t", na_rep="")

    (OUT / "signals_report.html").write_text(
        f"<html><head><meta charset='utf-8'>{css}</head><body><h2>Signal-Setups im Backtest</h2><ul>"
        + "".join(f"<li>{n}</li>" for n in notes) + "</ul>"
        + fig.to_html(full_html=False, include_plotlyjs="cdn")
        + "<h3>Kennzahlen (21 Coins)</h3>" + tbl(tab[tab.Coins == "21 Coins"][main_cols])
        + "<h3>Zufalls-Vergleich</h3>" + fig2.to_html(full_html=False, include_plotlyjs=False) + tbl(q)
        + "<p>Alle Zufallsläufe einzeln: <code>signals_random_runs.csv</code></p>"
        + "<h3>Alle Kennzahlen</h3>" + tbl(tab)
        + "</body></html>", encoding="utf-8")
    print(f"\nReport: {OUT / 'signals_report.html'}")


if __name__ == "__main__":
    main()
