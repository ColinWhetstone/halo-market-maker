"""
Sentiment-signal backtest on ALGO daily price data.

Entry rules (both must hold to go long, flat otherwise):
  - total_posts  2 days ago > its rolling 14-day mean
  - sentiment_ratio 3 days ago > 0.80

Costs        : 0.1% per round-trip trade (0.05% each leg, deducted on entry/exit)
Signals from : algo_sentiment_daily.csv
Price data   : algo_history.csv (1-min OHLCV, resampled to daily close)
Output       : sentiment_backtest.png
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── Load & resample price data ────────────────────────────────────────────────

price_raw = pd.read_csv("algo_history.csv")
price_raw["timestamp"] = pd.to_datetime(price_raw["timestamp"], unit="ms", utc=True)
price_raw = price_raw.set_index("timestamp").sort_index()

daily_close = price_raw["close"].resample("1D").last().dropna()
daily_close.index = pd.to_datetime(daily_close.index.date)
daily_close.index.name = "date"

price_df = pd.DataFrame({"close": daily_close})
price_df["daily_return"] = price_df["close"].pct_change()

# ── Load sentiment data ───────────────────────────────────────────────────────

sent_df = pd.read_csv("algo_sentiment_daily.csv", parse_dates=["date"])
sent_df = sent_df.set_index("date").sort_index()
sent_df.index = pd.to_datetime(sent_df.index)

# ── Build signal on sentiment index, then join to price ──────────────────────

# Rolling 14-day mean of total_posts (requires >=7 days to avoid noise)
sent_df["posts_ma14"] = (
    sent_df["total_posts"]
    .rolling(window=14, min_periods=7)
    .mean()
)

# Raw condition columns (shift applied after join so indices stay aligned)
sent_df["posts_above_ma"] = (sent_df["total_posts"] > sent_df["posts_ma14"]).astype(int)
sent_df["sent_above_08"]  = (sent_df["sentiment_ratio"] > 0.80).astype(int)

# ── Merge price + sentiment ───────────────────────────────────────────────────

df = price_df.join(
    sent_df[["posts_above_ma", "sent_above_08"]],
    how="left"
)

# Forward-fill sentiment on days with no Reddit data (weekends/gaps)
df[["posts_above_ma", "sent_above_08"]] = (
    df[["posts_above_ma", "sent_above_08"]].ffill().fillna(0)
)

# Apply the research lags:
#   total_posts signal at t-2  => shift signal forward by 2 so it aligns to t
#   sentiment_ratio at t-3     => shift forward by 3
df["posts_signal_lagged"]  = df["posts_above_ma"].shift(2)
df["sent_signal_lagged"]   = df["sent_above_08"].shift(3)

# Combined long signal: both conditions true
df["long_signal"] = (
    (df["posts_signal_lagged"] == 1) &
    (df["sent_signal_lagged"]  == 1)
).astype(int)

# Position is determined at close of signal day, enters next-day open
# (approximated here as close-to-close; position held at start of day t+1)
df["position"] = df["long_signal"].shift(1).fillna(0)

# Drop warmup rows before both signals are available
df = df.dropna(subset=["daily_return", "posts_signal_lagged", "sent_signal_lagged"])

# ── Transaction costs ─────────────────────────────────────────────────────────
# 0.1% round-trip = 0.05% per leg, charged on entry day and exit day

COST_PER_LEG = 0.0005   # 0.05%

position_change = df["position"].diff().fillna(df["position"])
# entry: position goes 0->1  (cost on that day's return)
# exit : position goes 1->0  (cost on that day's return)
df["trade_cost"] = position_change.abs() * COST_PER_LEG

# ── Strategy returns (net of costs) ──────────────────────────────────────────

df["strat_return_gross"] = df["position"] * df["daily_return"]
df["strat_return"]       = df["strat_return_gross"] - df["trade_cost"]

# Equity curves (start at 1.0)
df["bh_equity"]    = (1 + df["daily_return"]).cumprod()
df["strat_equity"] = (1 + df["strat_return"]).cumprod()

# ── Performance metrics ───────────────────────────────────────────────────────

total_return   = df["strat_equity"].iloc[-1] - 1
bh_return      = df["bh_equity"].iloc[-1] - 1
n_days         = len(df)

# Annualised Sharpe (252 trading days, risk-free = 0)
def sharpe_from_returns(rets: np.ndarray) -> float:
    mu, sigma = rets.mean(), rets.std(ddof=1)
    return float(mu / sigma * np.sqrt(252)) if sigma > 0 else np.nan

sharpe = sharpe_from_returns(df["strat_return"].values)

# Trade count: each rising edge of position = 1 trade entry
entries     = (df["position"].diff() == 1).sum()
days_in_mkt = int(df["position"].sum())
pct_in_mkt  = days_in_mkt / n_days * 100
total_cost  = df["trade_cost"].sum()

# Max drawdown
rolling_max = df["strat_equity"].cummax()
drawdown    = (df["strat_equity"] - rolling_max) / rolling_max
max_dd      = drawdown.min()

# ── Permutation test ──────────────────────────────────────────────────────────

N_PERMS   = 1000
rng       = np.random.default_rng(42)
raw_signal = df["long_signal"].values.copy()
returns    = df["daily_return"].values.copy()
costs      = df["trade_cost"].values.copy()   # costs depend on position edges,
                                              # recomputed per shuffle below

perm_sharpes: list[float] = []

for _ in range(N_PERMS):
    shuffled_signal = rng.permutation(raw_signal)
    # position: enter next day after signal
    perm_pos = np.roll(shuffled_signal, 1)
    perm_pos[0] = 0

    # recompute costs for this shuffle
    pos_series  = pd.Series(perm_pos, dtype=float)
    perm_cost   = pos_series.diff().abs().fillna(pos_series.abs()) * COST_PER_LEG

    perm_rets = perm_pos * returns - perm_cost.values
    perm_sharpes.append(sharpe_from_returns(perm_rets))

perm_sharpes_arr = np.array(perm_sharpes)
pct_beat = float(np.mean(perm_sharpes_arr > sharpe) * 100)
p_value  = pct_beat / 100   # one-sided permutation p-value

# ── Print report ──────────────────────────────────────────────────────────────

print("=" * 44)
print("  ALGO Sentiment Backtest Results")
print("=" * 44)
print(f"  Period          : {df.index[0].date()} to {df.index[-1].date()}")
print(f"  Days in sample  : {n_days}")
print(f"  Days in market  : {days_in_mkt}  ({pct_in_mkt:.1f}%)")
print(f"  Number of trades: {entries}")
print(f"  Total costs paid: {total_cost:.2%}")
print("-" * 44)
print(f"  Total return    : {total_return:+.2%}")
print(f"  Buy-&-hold      : {bh_return:+.2%}")
print(f"  Sharpe ratio    : {sharpe:.3f}")
print(f"  Max drawdown    : {max_dd:.2%}")
print("-" * 44)
print(f"  Permutation test ({N_PERMS} shuffles)")
print(f"  Random Sharpe   : {perm_sharpes_arr.mean():.3f} mean, "
      f"{perm_sharpes_arr.std():.3f} std")
print(f"  Shuffles beating actual Sharpe: {pct_beat:.1f}%")
print(f"  Permutation p-value           : {p_value:.3f}")
if p_value < 0.05:
    print("  Result: signal likely NOT due to luck (p < 0.05)")
elif p_value < 0.10:
    print("  Result: marginal evidence against luck (p < 0.10)")
else:
    print("  Result: cannot rule out luck (p >= 0.10)")
print("=" * 44)

# ── Plot ──────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(14, 9))
gs  = fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.35, wspace=0.3)

ax1 = fig.add_subplot(gs[0, :])   # equity curves — full width
ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)  # drawdown
ax3 = fig.add_subplot(gs[1, 1])   # permutation histogram

fig.suptitle("ALGO Sentiment Signal Backtest", fontsize=13)

# Top: equity curves
ax1.plot(df.index, df["strat_equity"], color="#1f77b4", linewidth=1.6,
         label=f"Strategy (net costs)  {total_return:+.1%}")
ax1.plot(df.index, df["bh_equity"],   color="#d62728", linewidth=1.2,
         linestyle="--", alpha=0.8, label=f"Buy & hold  {bh_return:+.1%}")

in_market = df["position"].astype(bool)
for start, end in zip(
    df.index[in_market & ~in_market.shift(1, fill_value=False)],
    df.index[in_market & ~in_market.shift(-1, fill_value=False)],
):
    ax1.axvspan(start, end, alpha=0.08, color="#1f77b4")

ax1.set_ylabel("Equity (normalised)", fontsize=10)
ax1.legend(fontsize=9)
ax1.grid(axis="y", linestyle="--", alpha=0.4)

# Bottom-left: drawdown
ax2.fill_between(df.index, drawdown.values, 0,
                 color="#d62728", alpha=0.5, label="Drawdown")
ax2.set_ylabel("Drawdown", fontsize=10)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax2.legend(fontsize=9)
ax2.grid(axis="y", linestyle="--", alpha=0.4)

ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax2.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
fig.autofmt_xdate(rotation=30, ha="right")

# Bottom-right: permutation Sharpe distribution
ax3.hist(perm_sharpes_arr, bins=40, color="#aec7e8", edgecolor="white",
         linewidth=0.4, label=f"Shuffled Sharpe (n={N_PERMS})")
ax3.axvline(sharpe, color="#1f77b4", linewidth=2,
            label=f"Actual Sharpe = {sharpe:.3f}")
ax3.axvline(np.percentile(perm_sharpes_arr, 95), color="grey",
            linewidth=1, linestyle="--", label="95th pct (random)")
ax3.set_xlabel("Sharpe Ratio", fontsize=10)
ax3.set_ylabel("Count", fontsize=10)
ax3.set_title(f"Permutation Test  (p = {p_value:.3f})", fontsize=10)
ax3.legend(fontsize=8)
ax3.grid(axis="y", linestyle="--", alpha=0.4)

plt.savefig("sentiment_backtest.png", dpi=150, bbox_inches="tight")
print("Plot saved to sentiment_backtest.png")
