"""Download and cache market data (public endpoints, no API key).

- 1m candles: Binance USDT-M futures (fallback: Binance spot)
- Funding history: hyperliquid | lighter | binance | constant | none
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).parent
CACHE = ROOT / "data"
CACHE.mkdir(exist_ok=True)
REPORTS = ROOT / "reports"


def report_dir(topic: str) -> Path:
    """Output folder for one topic, e.g. reports/liquidity."""
    p = REPORTS / topic
    p.mkdir(parents=True, exist_ok=True)
    return p
SESSION = requests.Session()
SESSION.headers["User-Agent"] = "grid-backtest/1.0"

MINUTE = 60_000
HOUR = 3_600_000
LIGHTER_MARKET_ID = {"ETH": 0, "BTC": 1}
HL_NAME = {"1000PEPE": "kPEPE", "1000SHIB": "kSHIB", "1000BONK": "kBONK", "1000FLOKI": "kFLOKI"}
_LIGHTER_IDS: dict | None = None


def lighter_market_id(coin: str) -> int | None:
    """Lighter market id for a Binance base name (None if not listed)."""
    global _LIGHTER_IDS
    if _LIGHTER_IDS is None:
        ob = SESSION.get("https://mainnet.zklighter.elliot.ai/api/v1/orderBooks", timeout=30).json()["order_books"]
        _LIGHTER_IDS = {o["symbol"]: int(o["market_id"]) for o in ob}
    for name in (coin, coin.replace("1000", ""), "k" + coin.replace("1000", "")):
        if name in _LIGHTER_IDS:
            return _LIGHTER_IDS[name]
    return None


def _get(url: str, params: dict):
    for attempt in range(5):
        try:
            r = SESSION.get(url, params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            if 400 <= r.status_code < 500:  # e.g. unknown symbol -> do not retry
                r.raise_for_status()
            r.raise_for_status()
            return r.json()
        except requests.HTTPError:
            raise
        except requests.RequestException:
            if attempt == 4:
                raise
            time.sleep(2 * (attempt + 1))


def _post(url: str, body: dict):
    for attempt in range(5):
        try:
            r = SESSION.post(url, json=body, timeout=30)
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if attempt == 4:
                raise
            time.sleep(2 * (attempt + 1))


# --------------------------------------------------------------------------- candles
INTERVAL_MS = {"1m": MINUTE, "1h": HOUR, "4h": 4 * HOUR, "1d": 24 * HOUR}
STABLES = {"USDC", "FDUSD", "TUSD", "BUSD", "DAI", "USDP", "USDE", "USD1", "RLUSD", "PYUSD", "XUSD", "EUR"}


def perp_universe() -> pd.DataFrame:
    """All Binance USDT-M crypto perps incl. delisted ones (status SETTLING). Columns: coin, status, delist_ms."""
    path = CACHE / "binance_perps.pkl"
    if path.exists() and time.time() - path.stat().st_mtime < 86400:
        return pd.read_pickle(path)
    syms = _get("https://fapi.binance.com/fapi/v1/exchangeInfo", {})["symbols"]
    rows = []
    for s in syms:
        if (s["contractType"] == "PERPETUAL" and s["quoteAsset"] == "USDT" and s.get("underlyingType") == "COIN"
                and s["baseAsset"] not in STABLES and s["status"] in ("TRADING", "SETTLING")):
            delist = int(s.get("deliveryDate") or 0)
            rows.append((s["symbol"][:-4], s["status"], delist if s["status"] == "SETTLING" else 0))
    df = pd.DataFrame(rows, columns=["coin", "status", "delist_ms"])
    df.to_pickle(path)
    return df


def load_klines(coin: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """OHLC candles of any interval from Binance USDT-M futures. Empty frame if symbol unknown."""
    step = INTERVAL_MS[interval]
    start_ms -= start_ms % step
    end_ms -= end_ms % step
    path = CACHE / f"{coin}_{interval}_{start_ms}_{end_ms}.pkl"
    if path.exists():
        return pd.read_pickle(path)
    rows, t = [], start_ms
    while t < end_ms:
        try:
            batch = _get("https://fapi.binance.com/fapi/v1/klines",
                         {"symbol": f"{coin}USDT", "interval": interval, "startTime": t,
                          "endTime": end_ms - 1, "limit": 1000 if interval == "1d" else 1500})
        except requests.HTTPError:
            batch = []
        if not batch:
            break
        rows.extend(batch)
        t = batch[-1][0] + step
    df = pd.DataFrame(rows, columns=range(12))[[0, 1, 2, 3, 4, 7]] if rows else pd.DataFrame(columns=range(6))
    df.columns = ["t", "open", "high", "low", "close", "quote_vol"]
    df = df.astype({"t": "int64", "open": float, "high": float, "low": float, "close": float, "quote_vol": float})
    df = df.drop_duplicates("t").sort_values("t").reset_index(drop=True)
    df.to_pickle(path)
    return df


def load_candles(coin: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """1m OHLC candles in [start_ms, end_ms). Cached as pickle."""
    start_ms -= start_ms % MINUTE
    end_ms -= end_ms % MINUTE
    path = CACHE / f"{coin}_1m_{start_ms}_{end_ms}.pkl"
    if path.exists():
        return pd.read_pickle(path)

    symbol = f"{coin}USDT"
    sources = [
        "https://fapi.binance.com/fapi/v1/klines",
        "https://data-api.binance.vision/api/v3/klines",
    ]
    last_err = None
    for url in sources:
        try:
            rows, t = [], start_ms
            while t < end_ms:
                batch = _get(url, params={"symbol": symbol, "interval": "1m", "startTime": t,
                                          "endTime": end_ms - 1, "limit": 1000 if "v3" in url else 1500})
                if not batch:
                    break
                rows.extend(batch)
                t = batch[-1][0] + MINUTE
                print(f"\r  {coin} candles: {len(rows):,}", end="", flush=True)
            print()
            break
        except requests.RequestException as e:  # try next source
            last_err = e
            print(f"\n  {url} failed ({e}), trying fallback ...")
    else:
        raise RuntimeError(f"No candle source reachable: {last_err}")

    df = pd.DataFrame(rows, columns=range(12))[[0, 1, 2, 3, 4]]
    df.columns = ["t", "open", "high", "low", "close"]
    df = df.astype({"t": "int64", "open": float, "high": float, "low": float, "close": float})
    df = df.drop_duplicates("t").sort_values("t").reset_index(drop=True)
    df.to_pickle(path)
    return df


# --------------------------------------------------------------------------- funding
def load_funding(coin: str, source: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Returns DataFrame[t, rate] with rate as fraction per settlement (long pays if > 0).

    source: 'hyperliquid' | 'lighter' | 'binance' | 'none' | '<float>' (= % per hour, constant)
    """
    try:
        pct_per_hour = float(source)
    except ValueError:
        pct_per_hour = None
    if pct_per_hour is not None:
        ts = range(start_ms - start_ms % HOUR + HOUR, end_ms, HOUR)
        return pd.DataFrame({"t": list(ts), "rate": pct_per_hour / 100})
    if source == "none":
        return pd.DataFrame({"t": pd.Series(dtype="int64"), "rate": pd.Series(dtype=float)})

    path = CACHE / f"{coin}_funding_{source}_{start_ms}_{end_ms}.pkl"
    if path.exists():
        return pd.read_pickle(path)

    rows: list[tuple[int, float]] = []
    if source == "hyperliquid":
        t = start_ms
        while t < end_ms:
            batch = _post("https://api.hyperliquid.xyz/info",
                          {"type": "fundingHistory", "coin": HL_NAME.get(coin, coin), "startTime": t, "endTime": end_ms})
            time.sleep(2.0)  # weight ~45 per call, limit 1200/min
            if not batch:
                break
            rows += [(int(x["time"]), float(x["fundingRate"])) for x in batch]
            t = int(batch[-1]["time"]) + 1
    elif source == "lighter":
        mid = lighter_market_id(coin)
        if mid is None:
            raise ValueError(f"{coin} not listed on Lighter")
        t = start_ms // 1000
        end_s = end_ms // 1000
        while t < end_s:
            chunk_end = min(end_s, t + 500 * 3600)
            res = _get("https://mainnet.zklighter.elliot.ai/api/v1/fundings",
                       {"market_id": mid, "resolution": "1h", "start_timestamp": t,
                        "end_timestamp": chunk_end, "count_back": 500})
            for x in res.get("fundings", []):
                sign = 1.0 if x.get("direction", "long") == "long" else -1.0
                rows.append((int(x["timestamp"]) * 1000, sign * float(x["rate"]) / 100))
            t = chunk_end
    elif source == "binance":
        t = start_ms
        while t < end_ms:
            try:
                batch = _get("https://fapi.binance.com/fapi/v1/fundingRate",
                             {"symbol": f"{coin}USDT", "startTime": t, "endTime": end_ms, "limit": 1000})
            except requests.HTTPError:
                batch = []
            if not batch:
                break
            rows += [(int(x["fundingTime"]), float(x["fundingRate"])) for x in batch]
            t = int(batch[-1]["fundingTime"]) + 1
            time.sleep(0.4)  # endpoint limit: 500 requests / 5 min / IP
    else:
        raise ValueError(f"unknown funding source: {source}")

    df = pd.DataFrame(rows, columns=["t", "rate"]).drop_duplicates("t").sort_values("t").reset_index(drop=True)
    df.to_pickle(path)
    return df
