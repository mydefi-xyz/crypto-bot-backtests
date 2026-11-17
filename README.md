# Crypto Bot Backtests

Grid, momentum rotation, signal setups, funding rate farming/arbitrage and copy trading – re-implemented in Python and tested on real historical market data, after fees, slippage and funding.

This repository contains the code and reports behind the article series **"Crypto trading bots backtested"** on MyDeFi:

- English: https://mydefi.xyz/learn/en/trading-bots-backtested
- German: https://mydefi.xyz/learn/de/trading-bots-im-test

[Deutsche Version dieser README](README.de.md)

> **Not investment advice.** Backtests rely on assumptions (fees, execution, data source) and say nothing certain about the future. Use at your own risk.

## Results at a glance

| Bot | Tested | Result after costs | Reliable? |
|---|---|---|---|
| **Grid bot** (liquidity range / V3 style) | BTC, ETH, 2 × 6 months, 141 settings in 570 runs | median −4 % per 6 months, 0 of 120 sweep settings positive in all four runs | no |
| **Momentum rotation** (long/short) | 10–650 coins, 2 × 12 months, 1-day execution lag | previous year +69 % to +163 %, last 12 months −50 % to +11 % (2× leverage), drawdowns 44–57 % | no, regime-dependent |
| **Signal setups** (ATH, EMA, MACD+RSI, volume profile, breakout) | 21 coins, 24 months, 4,315 trades | all five negative (−22 R to −141 R), none better than random (best p = 0.27) | no |
| **Funding arbitrage** (dynamic) | 25 coins, 3 exchanges, 2 × 12 months | −11.7 % to +0.5 % p.a. | no |
| **Funding farming** (static, spot hedge) | BTC/ETH and 20 large caps | +0.3 % to +10.7 % p.a., BTC/ETH on Hyperliquid +3.6 % to +9.3 %, account drawdown ≤ 1 % | yes, with exchange risk |
| **Funding farming** (two perps) | short Hyperliquid, long Binance | −6.5 % / +3.4 % p.a. (liquidations in the crash of 10 Oct 2025) | no |
| **Copy trading** (Hyperliquid) | 186 accounts, 487 observations, 90/90 days | median time-weighted return −12 % to −26 % in the 90 days after selection | no |

Provider fees (subscription, fee per fill, profit share) are only included where stated. They make every result worse.

**Status of the rotation runs:** the runs that need the live Hyperliquid coin list (`--universe hyperliquid`, `top20vol180`, `top40vol365`) still contain results from before the latest fixes (execution lag, listing dates). Re-run them with the commands below once the Hyperliquid API is reachable.

## Installation

```bash
git clone https://github.com/mydefi-xyz/crypto-bot-backtests.git
cd crypto-bot-backtests
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.14, pandas 3.1, numpy 2.5, requests 2.32, plotly 7.1 (Python ≥ 3.11 should work). No API key is needed: all data comes from public endpoints and is cached in `data/` (not in the repository, roughly 150 MB). The first run of a script downloads its data and takes correspondingly longer.

Code comments, console output and report labels are partly in German.

## Structure

```
.
├── data.py                 # data loaders + cache (candles, funding, universes), report_dir()
├── engine.py               # grid bot simulator (1-minute candles)
├── run.py                  # grid bot: 3 variants, report with grid boxes
├── sweep.py                # grid bot: parameter sweep (120 combinations per coin and period)
├── sweep_exit.py           # grid bot: early close at X % of the range
├── explore_check.py        # re-calculation of typical APR leaderboard setups incl. inventory
├── rotation.py             # momentum rotation, simple variant (fixed coin list, 1h)
├── rotation_xs.py          # momentum rotation, reference setup (tranches, 40 markets, delisted coins)
├── rotation_xs_summary.py  # comparison table of saved rotation_xs runs
├── signals.py              # 5 signal setups + random-direction baseline
├── funding_download.py     # download funding history Binance / Hyperliquid / Lighter + candles
├── funding.py              # funding farming vs. funding arbitrage
├── copytrade.py            # copy trading test with real Hyperliquid accounts
├── tools_slim_reports.py   # helper: replace embedded plotly.js in HTML reports by a CDN link
└── reports/
    ├── liquidity/   rotation/   signals/   funding/   copytrade/
```

All reports are written to `reports/<topic>/`. HTML reports are interactive (Plotly via CDN). GitHub does not render HTML: download the file or open it via `https://htmlpreview.github.io/?<raw-URL>`.

## Scripts

### 1. Grid bot (liquidity range)

| Command | What it does | Output |
|---|---|---|
| `python run.py --end 2026-11-06T10:31` | BTC + ETH, 180 days, variant 1 (long grid with provider fee), 2 (symmetric + stop & restart), 3 (symmetric + re-centering) | `liquidity_report_<tag>.html`, `liquidity_results_<tag>.csv`, `liquidity_grids_<tag>.csv` |
| `python run.py --coins BTC --start 2026-11-01T20:05 --end 2026-11-05T20:23 --pmin 74990 --pmax 88031 --pool 2501 --variants 1,2 --report calibration_live_bot_nov.html` | calibration against a real live bot | `calibration_live_bot_nov.html` + tagged CSVs |
| `python sweep.py` / `python sweep.py --end 2026-05-10T10:31` | 120 combinations (range, level spacing, sideways filter) for two 6-month periods | `sweep_YYYYMMDD.csv` |
| `python sweep_exit.py` | close at 20/30/50 % of the range, restart immediately vs. when sideways | `sweep_exit.csv` |
| `python explore_check.py` | setups from a typical APR leaderboard re-calculated incl. inventory movement | `explore_check.csv` |

Main options of `run.py` (see `--help`):

| Option | Default | Meaning |
|---|---|---|
| `--coins` | `BTC,ETH` | coins |
| `--days` / `--start` / `--end` | 180 / – / now | period (UTC) |
| `--pool` | 2000 | budget in USD |
| `--step` | 0.30 | level spacing ("fee rate") in % |
| `--range` | 10 | half range width in % |
| `--exchange` | `extended` | fee preset (`extended`, `hyperliquid`, `lighter`, `binance`) |
| `--maker` / `--taker` | preset | override maker / taker fee in % |
| `--builder` | 0.02 | provider fee (builder fee) per fill in % (variant 1 only) |
| `--funding` | `hyperliquid` | `hyperliquid`, `lighter`, `binance`, `none` or constant %/h |
| `--slippage` | 0.05 | slippage on market/stop orders in % |
| `--stop-buffer` / `--max-loss` | 0.5 / 2.0 | stop outside the range / loss limit per grid (%) |
| `--cooldown` | 4 | minimum wait after a stop in hours (variants 2+3) |
| `--calm` / `--calm-hours` | 3.0 / 12 | restart only when high-low of the last x h < y % |
| `--recenter` | 50 | variant 3: re-center at x % of the half range |
| `--tag` | derived | suffix for output files, so different runs do not overwrite each other |

### 2. Momentum rotation

| Command | What it does |
|---|---|
| `python rotation.py` | simple variant: 15 coins, 1h candles, lookback 7 days, rebalancing every 8 h, plus previous year |
| `python rotation.py --rebalance 72` | slow variant (output files `*_rb72.*`) |
| `python rotation_xs.py --universe hyperliquid` | reference setup: 40 most liquid coins (30-day volume), lookback 14 days, top/flop 20 %, 3 tranches × 3 days, 2× leverage |
| `python rotation_xs.py --universe hyperliquid --no-funding` | same without funding |
| `python rotation_xs.py --no-funding` | all Binance perps (incl. delisted) without funding (tag `binance_nofunding`) |
| `python rotation_xs.py --universe hyperliquid --coins BTC,ETH,SOL,XRP,BNB,DOGE,ADA,LINK,AVAX,SUI,LTC,BCH,TRX,DOT,TON,XLM,HBAR,NEAR,UNI,AAVE --markets 20 --tag majors20` | 20 large caps only |
| `python rotation_xs.py --universe hyperliquid --coins BTC,ETH,SOL,XRP,BNB,DOGE,ADA,LINK,AVAX,LTC --markets 10 --tag majors10` | 10 large caps |
| `python rotation_xs.py --coins <the 20 above>,1000SHIB,1000PEPE,APT,ICP,ETC,ONDO,TAO,ATOM,FIL,ARB,OP,INJ,RENDER,FET,SEI,TIA,ENA,WLD,ALGO,XMR --markets 40 --tag majors40` | 40 large caps (Binance universe) |
| `python rotation_xs.py --universe hyperliquid --markets 20 --liq-days 180 --tag top20vol180` | top 20 by 180-day volume |
| `python rotation_xs.py --universe hyperliquid --markets 40 --liq-days 365 --tag top40vol365` | top 40 by 365-day volume |
| `python rotation_xs_summary.py hyperliquid majors20` | compare saved runs |

Options of `rotation_xs.py`: `--lev` (leverage, 2), `--hold` (tranches / holding days, 3), `--lookback` (days, 14), `--markets` (40), `--cost` (% per side incl. provider fee, 0.09), `--universe` (`binance` / `hyperliquid`), `--coins`, `--liq-days` (30), `--exec-lag` (bars between ranking and execution, default 1; 0 reproduces the old same-close execution), `--no-funding`, `--skip-sweep`, `--tag`. `rotation.py` has `--exec-lag` too (hours, default 1).

Reports include a table with execution lags of 0/1/2 bars, turnover in USD and as a multiple of average equity, and per-coin statistics (`rotation_xs_coins_<tag>.csv`). The Hyperliquid coin list is cached in `data/hl_meta_coins.json`; a coin is only tradable from its first Hyperliquid record.

### 3. Signal setups

`python signals.py` – ATH rejection, EMA pullback, MACD + RSI, volume profile (4h) and breakout (1d) on 21 coins over 24 months, incl. Binance funding and a comparison with random direction (50 runs, same entries, stops and targets). Output: `signals_report.html`, `signals_results.csv`, `signals_trades.csv`, `signals_random_runs.csv`. Method details: [reports/signals/README.md](reports/signals/README.md).

### 4. Funding farming and arbitrage

```bash
python funding_download.py   # once, ~30–40 min (Hyperliquid rate-limits)
python funding.py            # --pool 2000 --lev 2 --k 5 --rebalance 8 --lookback 72 --min-apr 15
```

Margin of the perp legs is simulated on 1h candles: daily check at 00:00 UTC, rebalancing to the target leverage when the price moved more than `--trigger` % (25) or the margin ratio fell below `--min-margin` % (15), liquidation when the hourly high/low reaches the liquidation price (`--mmr`, 1 % maintenance margin). Output: `funding_report.html`, `funding_results.csv`, `funding_sweep.csv`, `funding_avg.csv`, `funding_liquidations.csv`.

### 5. Copy trading

`python copytrade.py --n 800` – random sample of active Hyperliquid accounts, selection on 90 days, measurement in the following 90 days (4 windows). Default metric is the time-weighted return from the portfolio history (deposits do not count as gains). Options: `--offline` (cache only), `--refresh` (reload leaderboard), `--no-deposit-adj` (strict variant without deposit adjustment, separate output files), `--form` / `--test` (window lengths), `--cost`, `--min-active`. Output: `copytrade_results.csv`, `copytrade_obs.csv`, `copytrade_excluded.csv`.

The leaderboard is a snapshot; a run on another day gives slightly different numbers.

## Data sources

| Data | Source |
|---|---|
| Candles 1m / 1h / 4h / 1d | Binance USDT-M futures (`fapi.binance.com`), fallback Binance spot |
| Perp universe incl. delisted coins | Binance `exchangeInfo` (status `SETTLING`), Hyperliquid `meta` |
| Funding | Binance `fundingRate`, Hyperliquid `fundingHistory`, Lighter `fundings` |
| Trader accounts | Hyperliquid leaderboard (`stats-data.hyperliquid.xyz`) + `portfolio` |

## Assumptions and limitations

- **Prices:** Binance candles stand in for DEX prices (difference mostly < 0.1 %).
- **Grid execution:** a limit order only fills when the price crosses the level. Intrabar path: green O→L→H→C, red O→H→L→C. Stops, loss limit and early close are checked against the intrabar extreme and filled at the worse of trigger price and extreme, plus slippage.
- **Fees:** presets per exchange (base tier; Extended maker 0 % / taker 0.025 %), slippage 0.03–0.05 % on market/stop orders. Provider fee 0.02 % per fill only in grid variant 1 and in the rotation costs.
- **Rotation:** daily closes, execution one bar after the ranking, flat costs per side, no volatility-based leverage control, liquidity from Binance volume. Coin lists and variants were chosen on the same two years they were tested on – the previous year is not a true out-of-sample period. Delisted Binance coins are only included if they still appear with status `SETTLING`.
- **Signals:** coins selected by today's size (survivorship), prices and funding from Binance. Trades run partly in parallel; significance is computed on weekly sums.
- **Funding:** the basis (price difference spot/perp or between exchanges) and margin transfers between exchanges are not modelled; margin checks are daily.
- **Copy trading:** only accounts that are still on today's leaderboard (survivorship) → results rather too optimistic. Portfolio history is weekly; copy costs are estimated from lifetime turnover because window volume is not available from the cache.
- **Out-of-sample:** where possible, every setting was also checked on a second period.

## License

[MIT](LICENSE) © MyDeFi
