"""Grid bot / liquidity-range simulator on 1m candles.

Modelled like a typical commercial grid ladder:
- geometric levels L_i = Pmin * (1+step)^i up to Pmax
- constant coin quantity q per level, q = usable_budget / sum(L_i)
  (reproduces the minimum-order-size check of a live reference bot exactly)
- state = anchor index a: a resting buy at L[a-1] and a resting sell at L[a+1].
  A fill moves the anchor by one level, the counter-order is placed immediately.

mode "long": standard long grid. Buys the inventory for every level above the start price at start.
mode "sym" : symmetric grid. Starts flat, goes long below start, short above start
             (= long grid + static short hedge, but on one account).

Fill model: limit order fills only if price trades THROUGH the level (strictly).
Intra-candle path: green candle O->L->H->C, red candle O->H->L->C.
Stops, loss limit and early close are checked at every path point (i.e. against the candle's
extremes) and fill at that point (= worse of trigger price and extreme) plus slippage.
Drawdown uses the worst intrabar equity. Funding is booked at the close of the minute that
ends at the funding timestamp.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Config:
    name: str
    mode: str                          # "long" | "sym"
    pool: float = 2000.0               # whole budget (USD)
    usable_frac: float = 0.9           # typical: 90% ladder, 10% buffer
    step: float = 0.003                # level spacing ("fee rate")
    range_pct: float = 0.10            # half width of range around start
    pmin: float | None = None          # explicit range (first grid only, later grids keep the ratios)
    pmax: float | None = None
    maker: float = 0.0
    taker: float = 0.00025
    builder: float = 0.0               # bot-provider fee per fill (e.g. 0.0002 = 0.02 %)
    slippage: float = 0.0005           # for market / stop orders
    stop_buffer: float | None = None   # stop distance outside range, e.g. 0.005. None = no stop
    max_loss: float | None = None      # loss limit per grid as fraction of pool. None = off
    restart: bool = False              # auto restart after stop (centered on new price)
    cooldown_min: int = 240            # wait at least this long before restart
    calm_window: int = 720             # sideways detection window in minutes
    calm_range: float = 0.03           # restart only if (high - low)/price over calm_window < this
    recenter_drift: float | None = None  # recenter if price moved >= this fraction of half range
                                         # AND the grid's total pnl covers the closing cost
    exit_drift: float | None = None    # close unconditionally at this fraction of half range (e.g. 0.3)
    exit_restart_now: bool = True      # after exit_drift: restart at the same minute's close (else cooldown + calm)
    size_mult: float = 1.0
    min_qty: float = 0.0


@dataclass
class Result:
    cfg: Config
    metrics: dict
    equity: np.ndarray
    events: list = field(default_factory=list)
    grids: list = field(default_factory=list)   # one dict per grid lifetime


class _Sim:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.mf = cfg.maker + cfg.builder
        self.tf = cfg.taker + cfg.builder
        self.cash = cfg.pool
        self.pos = 0.0
        self.fees = self.funding = self.income = self.volume = 0.0
        self.fills = 0
        self.counts = {"start": 0, "stop": 0, "loss": 0, "recenter": 0, "exit": 0}
        self.events: list[tuple[int, str, float]] = []
        self.active = False
        self.inactive_since = 0
        self.lo_ratio = 1 - cfg.range_pct
        self.hi_ratio = 1 + cfg.range_pct
        self.first = True
        self.warn_qty = False
        self.grids: list[dict] = []
        self.cur: dict | None = None

    def end_grid(self, i: int, reason: str, equity: float):
        g = self.cur
        if g is None:
            return
        g.update(end=i, reason=reason,
                 fees_net=(self.income - g.pop("_inc")) - (self.fees - g.pop("_fee")),
                 funding=self.funding - g.pop("_fund"),
                 fills=self.fills - g.pop("_fills"),
                 pnl=equity - g.pop("_eq0"))
        g["position"] = g["pnl"] - g["fees_net"] - g["funding"]
        self.grids.append(g)
        self.cur = None

    # ---------------------------------------------------------------- orders
    def trade(self, dq: float, px: float, fee: float):
        self.cash -= dq * px
        self.pos += dq
        notional = abs(dq) * px
        f = notional * fee
        self.cash -= f
        self.fees += f
        self.volume += notional
        self.fills += 1

    def close_all(self, px: float, i: int, reason: str):
        if abs(self.pos) > 1e-12:
            s = 1.0 if self.pos > 0 else -1.0
            self.trade(-self.pos, px * (1 - s * self.cfg.slippage), self.tf)
        self.pos = 0.0
        self.active = False
        self.inactive_since = i
        self.counts[reason] += 1
        self.events.append((i, reason, px))
        self.end_grid(i, reason, self.cash)

    def start_grid(self, price: float, i: int):
        c = self.cfg
        if self.first and c.pmin and c.pmax:
            self.lo_ratio, self.hi_ratio = c.pmin / price, c.pmax / price
        self.first = False
        lo, hi = price * self.lo_ratio, price * self.hi_ratio
        n = int(math.floor(math.log(hi / lo) / math.log1p(c.step))) + 1
        lv = lo * (1 + c.step) ** np.arange(n)
        self.lv = lv.tolist()
        self.N = n
        self.q = c.pool * c.usable_frac / lv.sum() * c.size_mult
        if self.q < c.min_qty:
            self.warn_qty = True
        self.a = self.a0 = int(np.argmin(np.abs(lv - price)))
        self.center = price
        self.grid_eq0 = self.cash + self.pos * price
        self.cur = dict(start=i, lo=lv[0], hi=lv[-1], center=price, _eq0=self.grid_eq0,
                        _inc=self.income, _fee=self.fees, _fund=self.funding, _fills=self.fills)
        if c.mode == "long":
            inv = self.q * (n - 1 - self.a0)
            if inv > 0:
                self.trade(inv, price * (1 + c.slippage), self.tf)
        if c.stop_buffer is not None:
            self.stop_lo = self.lv[0] * (1 - c.stop_buffer)
            self.stop_hi = self.lv[-1] * (1 + c.stop_buffer)
        self.active = True
        self.counts["start"] += 1
        self.events.append((i, "start", price))

    # ---------------------------------------------------------------- price path
    def move(self, p: float, target: float):
        lv, q, a0 = self.lv, self.q, self.a0
        if target < p:
            while self.a > 0 and lv[self.a - 1] > target:
                self.trade(q, lv[self.a - 1], self.mf)
                if self.a > a0:  # closes a short -> completed cycle
                    self.income += q * (lv[self.a] - lv[self.a - 1])
                self.a -= 1
        elif target > p:
            while self.a < self.N - 1 and lv[self.a + 1] < target:
                self.trade(-q, lv[self.a + 1], self.mf)
                if self.a < a0:  # closes a long -> completed cycle
                    self.income += q * (lv[self.a + 1] - lv[self.a])
                self.a += 1

    def check_triggers(self, px: float, i: int):
        """Early close, range stop and loss limit at an intrabar path point.

        px is the segment end (candle extreme, open on a gap, or close). Once a trigger is hit,
        px is already at or beyond the trigger price, so filling at px = worse of both (conservative).
        Order = order in which the triggers are reached on a monotone move (exit lies inside the range).
        """
        c = self.cfg
        if c.exit_drift is not None and abs(px / self.center - 1) >= c.exit_drift * (self.hi_ratio - 1):
            self.close_all(px, i, "exit")
        elif c.stop_buffer is not None and (px <= self.stop_lo or px >= self.stop_hi):
            self.close_all(px, i, "stop")
        elif c.max_loss is not None and self.cash + self.pos * px - self.grid_eq0 < -c.max_loss * c.pool:
            self.close_all(px, i, "loss")


def simulate(df: pd.DataFrame, funding: pd.DataFrame, cfg: Config) -> Result:
    t = df["t"].to_numpy()
    O, H, L, C = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    n = len(C)

    fund = np.zeros(n)
    if len(funding):
        # funding timestamps carry ms jitter -> round to the minute, then book on the candle
        # that closes at that time (position and price at the funding moment)
        ft = (funding["t"].to_numpy() + 30_000) // 60_000 * 60_000
        idx = np.searchsorted(t, ft, side="left") - 1
        ok = (idx >= 0) & (ft <= t[-1] + 60_000)
        np.add.at(fund, idx[ok], funding["rate"].to_numpy()[ok])

    w = max(1, cfg.calm_window)
    roll = ((pd.Series(H).rolling(w, min_periods=w).max()
             - pd.Series(L).rolling(w, min_periods=w).min()) / pd.Series(C)).fillna(1.0).to_numpy()

    s = _Sim(cfg)
    equity = np.empty(n)
    eq_low = np.empty(n)   # worst intrabar equity per minute
    in_range = 0
    s.start_grid(O[0], 0)
    last = O[0]

    for i in range(n):
        o, h, l, c = O[i], H[i], L[i], C[i]
        path = (o, l, h, c) if c >= o else (o, h, l, c)
        worst = math.inf
        for px in path:
            if s.active:
                s.move(last, px)
                s.check_triggers(px, i)
            last = px
            # equity is concave in price between path points (position falls as price rises)
            # -> segment endpoints give the intrabar minimum
            worst = min(worst, s.cash + s.pos * px)

        if fund[i] != 0.0 and s.pos != 0.0:
            pay = s.pos * c * fund[i]
            s.cash -= pay
            s.funding -= pay

        if s.active:
            if s.lv[0] <= c <= s.lv[-1]:
                in_range += 1
            gp = s.cash + s.pos * c - s.grid_eq0
            if (cfg.recenter_drift is not None
                    and abs(c / s.center - 1) >= cfg.recenter_drift * (s.hi_ratio - 1)
                    and gp - abs(s.pos) * c * (s.tf + cfg.slippage) >= 0):
                s.close_all(c, i, "recenter")
                s.start_grid(c, i)
        elif cfg.exit_drift is not None and cfg.exit_restart_now and s.events[-1][1] == "exit":
            s.start_grid(c, i)  # immediate restart after an early close
        elif cfg.restart and i - s.inactive_since >= cfg.cooldown_min and roll[i] < cfg.calm_range:
            s.start_grid(c, i)

        equity[i] = s.cash + s.pos * c
        eq_low[i] = min(worst, equity[i])

    if s.active:
        s.end_grid(n - 1, "offen", equity[-1])
    wins = sum(1 for g in s.grids if g["pnl"] > 0)

    days = (t[-1] - t[0]) / 86_400_000
    total = equity[-1] - cfg.pool
    # peak from closes before the minute, trough from intrabar worst equity
    peak = np.maximum.accumulate(np.concatenate(([cfg.pool], equity[:-1])))
    dd = np.minimum(eq_low - peak, equity - np.maximum.accumulate(equity))
    m = {
        "Netto-Ergebnis $": total,
        "Netto %": 100 * total / cfg.pool,
        "APR %": 100 * total / cfg.pool / days * 365,
        "Grid-Ertrag brutto $": s.income,
        "Gebuehren $": -s.fees,
        "Net collected fees $": s.income - s.fees,
        "Funding $": s.funding,
        "Positions-PnL $": total - s.income + s.fees - s.funding,
        "Max Drawdown $": dd.min(),
        "Max Drawdown %": 100 * dd.min() / cfg.pool,
        "Fills": s.fills,
        "Volumen $": s.volume,
        "Grids gestartet": s.counts["start"],
        "Stopps (Range)": s.counts["stop"],
        "Stopps (Verlustlimit)": s.counts["loss"],
        "Nachgezogen": s.counts["recenter"],
        "Grids mit Gewinn / gesamt": f"{wins} / {len(s.grids)}",
        "Zeit aktiv in Range %": 100 * in_range / n,
        "Qty/Level": s.q,
        "Position Ende": s.pos,
    }
    if s.warn_qty:
        m["WARNUNG"] = f"Qty/Level < Mindestgroesse {cfg.min_qty}"
    return Result(cfg, m, equity, s.events, s.grids)
