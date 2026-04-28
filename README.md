# ALGO/USD Market Making & Sentiment Research

Research and backtesting suite for the ALGO/USD pair on Kraken/KuCoin.
Three workstreams: **Avellaneda–Stoikov market making**, **Reddit sentiment analysis**, and data collection utilities.

---

## Project Structure

```
mm-bot/
├── Market Making
│   ├── as_engine.py          Core A-S pricing model
│   ├── competitor_floor.py   Spread floor estimation
│   ├── regime.py             Volatility regime classifier
│   └── strategy.py           Full backtest runner
│
├── Data Collection
│   ├── collect_data.py       Live tick data collector (Kraken)
│   ├── fetch_data.py         One-shot order book snapshot
│   └── download_history.py   Bulk OHLCV downloader (KuCoin)
│
├── Sentiment Analysis
│   ├── sentiment_collector.py   Scrapes Reddit, writes CSVs
│   ├── sentiment_analysis.py    Correlation study + chart
│   └── sentiment_backtest.py    Signal backtest + permutation test
│
└── Data Files
    ├── algo_data.csv            Live tick data (collected by collect_data.py)
    ├── algo_history.csv         90-day 1-min OHLCV (from download_history.py)
    ├── algo_sentiment.csv       Raw Reddit posts (from sentiment_collector.py)
    └── algo_sentiment_daily.csv Daily aggregated sentiment
```

---

## Avellaneda–Stoikov Market Making

### `as_engine.py`

Implements the closed-form Avellaneda–Stoikov (A–S) model primitives. No market logic here — pure math.

- **`calculate_reservation_price(mid, inventory, vol, gamma, tau)`** — Skews fair value away from mid based on inventory: `r = mid - q * gamma * sigma^2 * tau`. Long inventory pushes r below mid, short pushes it above.
- **`calculate_spread(vol, gamma, tau, kappa)`** — Computes the optimal total quoted spread: `s* = gamma * sigma^2 * tau + (2/gamma) * ln(1 + gamma/kappa)`. Widens under high volatility or risk aversion; tightens when fills decay quickly (high kappa).
- **`get_quotes(mid, inventory, vol, gamma, tau, kappa)`** — Returns `(bid, ask)` by centering the optimal spread around the reservation price.

```bash
python as_engine.py   # runs a self-test with example parameters
```

---

### `competitor_floor.py`

Estimates the tightest spread any competitor is currently quoting, used to prevent the strategy from posting quotes that are too wide to ever get filled.

- **`compute_competitor_floor(spreads, window=100)`** — Rolling minimum of observed market spreads over the last `window` ticks.
- **`compute_repricing_velocity(spreads, window=20)`** — Counts how often the spread changes within a rolling window. Higher velocity = more active competitors.

```bash
python competitor_floor.py   # prints last 10 floor + velocity estimates from algo_data.csv
```

---

### `regime.py`

Classifies each bar into one of three volatility regimes using rolling lag-1 return autocorrelation over a 500-observation window. Replaced an earlier Hurst estimator.

| Regime | Condition | Interpretation |
|--------|-----------|----------------|
| `high` | autocorr > 0.05 | Trending / momentum market |
| `low` | autocorr < -0.05 | Mean-reverting market |
| `normal` | otherwise | Neutral / warmup period |

The regime label feeds directly into `strategy.py` to adjust risk aversion (`gamma`).

```bash
python regime.py algo_data.csv   # prints regime distribution on any CSV with a mid_price column
```

---

### `strategy.py`

Full A–S market making backtest. Wires together `as_engine.py`, `competitor_floor.py`, and `regime.py` into a tick-by-tick simulation.

**Key parameters:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `KAPPA` | 150,000 | Market depth (fill intensity decay) |
| `INVENTORY_CAP` | 200 units | Hard position limit |
| `EMERGENCY_THRESH` | 150 units | Triggers 5x gamma widening |
| `FILL_QTY` | 100 units | Size per fill |
| `MIN_SPREAD_BPS` | 10 bps | Spread floor |
| `MAX_SPREAD_BPS` | 50 bps | Spread cap |
| `GAMMA` | low: 0.6, normal: 1.0, high: 2.5 | Risk aversion by regime |
| `WARMUP_ROWS` | 500 | Bars skipped before trading begins |

**Fill model:**
- *Tick data*: fill triggered when `last_trade_price` crosses the quoted bid or ask level
- *OHLCV data*: probabilistic fill (10% chance per candle) when candle low/high touches quotes, with a 30-candle cooldown

**Outputs:** spread-capture PnL, inventory MTM PnL, Sharpe, fills per day, max drawdown. Compares dynamic (regime-adaptive) gamma vs static gamma=0.1 side-by-side.

```bash
python strategy.py                      # single run on algo_data.csv
python strategy.py --csv algo_history.csv
python strategy.py --monte-carlo        # 10-run Monte Carlo comparison
```

---

## Data Collection

### `collect_data.py`

Continuous live tick collector. Polls Kraken every 10 seconds via `ccxt`, writing one row per poll to `algo_data.csv`. Recovers automatically from errors with a 30-second backoff. Runs until interrupted.

**Output columns:** `timestamp`, `best_bid`, `best_ask`, `bid_volume`, `ask_volume`, `mid_price`, `spread`, `last_trade_price`, `last_trade_volume`

```bash
python collect_data.py   # runs indefinitely, Ctrl+C to stop
```

---

### `fetch_data.py`

One-shot snapshot of the ALGO/USD Kraken order book. Prints the top 5 bids and asks with price and volume. Useful for a quick sanity check before running the collector.

```bash
python fetch_data.py
```

---

### `download_history.py`

Downloads 90 days of 1-minute OHLCV for ALGO/USDT from KuCoin in paginated chunks of 720 candles. Overwrites `algo_history.csv` on each run. Includes rate-limit courtesy sleeps between requests.

**Output:** `algo_history.csv` — columns: `timestamp` (ms epoch), `open`, `high`, `low`, `close`, `volume`

```bash
python download_history.py   # takes a few minutes; ~130k rows
```

---

## Sentiment Analysis

### `sentiment_collector.py`

Scrapes Reddit for ALGO-related posts from two subreddits and writes two CSV files. Covers the last 90 days. Run this to refresh sentiment data before running the analysis or backtest.

**Sources:**
- `r/algorand` — latest new posts via `/new.json`
- `r/CryptoCurrency` — posts mentioning ALGO via `/search.json?q=ALGO&sort=new`

**Sentiment scoring:** `+1` if Reddit score > 0, `-1` if < 0, `0` if zero.

**Outputs:**

| File | Description |
|------|-------------|
| `algo_sentiment.csv` | Raw posts: `timestamp`, `source`, `title`, `sentiment_score` |
| `algo_sentiment_daily.csv` | Daily aggregates: `date`, `total_posts`, `avg_sentiment_score`, `positive_count`, `negative_count`, `neutral_count`, `sentiment_ratio` |

```bash
python sentiment_collector.py
```

> **Note:** CryptoPanic is blocked by Cloudflare for automated clients and has been removed from this script.

---

### `sentiment_analysis.py`

Loads `algo_sentiment_daily.csv` and `algo_history.csv`, resamples price to daily close, and runs a Pearson cross-correlation study between sentiment signals and next-day returns.

**What it does:**
1. Resamples 1-min OHLCV to daily last-close and computes daily returns
2. Inner-joins sentiment and price on date (72 aligned days)
3. Runs `scipy.stats.pearsonr` at lags 0, 1, 2, 3, 5, 7 days for two signals:
   - `sentiment_ratio` (positive posts / total posts)
   - `total_posts` (raw post volume)
4. Prints a significance table and saves a two-panel chart

**Key findings:**
- `sentiment_ratio` lag-3: r = -0.24, p = 0.045 (high positivity precedes slight pullback)
- `total_posts` lag-2: r = +0.24, p = 0.044 (higher volume precedes slight rally)

**Output:** `sentiment_analysis.png` — ALGO daily close price (top panel) and daily sentiment ratio (bottom panel) over the aligned date range.

```bash
python sentiment_analysis.py
```

---

### `sentiment_backtest.py`

Translates the two significant correlation findings into a simple long/flat signal and backtests it on daily ALGO returns. Includes transaction costs and a permutation significance test.

**Signal logic:**
- Go **long** when both conditions hold:
  - `total_posts` 2 days ago > its rolling 14-day mean
  - `sentiment_ratio` 3 days ago > 0.80
- Go **flat** otherwise

**Transaction costs:** 0.1% per round-trip (0.05% per leg, applied on entry and exit days).

**Permutation test:** shuffles the signal randomly 1,000 times, recomputes Sharpe each time, and reports what fraction of shuffles beat the actual Sharpe. p-value is the fraction that beat it (one-sided).

**Results (as of March 2026, 88-day window):**

| Metric | Value |
|--------|-------|
| Total return | +22.21% |
| Buy-and-hold | -16.71% |
| Sharpe ratio | 1.484 |
| Max drawdown | -17.11% |
| Trades | 14 |
| Days in market | 35 (39.8%) |
| Permutation p-value | 0.062 |

**Output:** `sentiment_backtest.png` — three panels: equity curve vs buy-and-hold (with shaded in-market periods), drawdown, and permutation test Sharpe distribution.

```bash
python sentiment_backtest.py
```

---

## Data Files

| File | Rows | Updated by | Description |
|------|------|------------|-------------|
| `algo_data.csv` | ~14,500 | `collect_data.py` | Live tick snapshots from Kraken, 10-second cadence |
| `algo_history.csv` | ~130,000 | `download_history.py` | 90-day 1-min OHLCV from KuCoin |
| `algo_sentiment.csv` | ~446 | `sentiment_collector.py` | Raw Reddit posts with sentiment scores |
| `algo_sentiment_daily.csv` | ~74 | `sentiment_collector.py` | Daily aggregated sentiment metrics |

---

## Dependencies

```bash
pip install ccxt pandas numpy scipy matplotlib requests
```

---

## Recommended Run Order

```bash
# 1. Refresh price history
python download_history.py

# 2. Refresh sentiment data
python sentiment_collector.py

# 3. Run correlation study
python sentiment_analysis.py

# 4. Run sentiment backtest
python sentiment_backtest.py

# 5. Run market making backtest
python strategy.py --csv algo_history.csv
python strategy.py --monte-carlo
```
