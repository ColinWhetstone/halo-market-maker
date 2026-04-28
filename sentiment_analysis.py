"""
Cross-correlation of Reddit sentiment signals vs ALGO next-day returns.

Inputs:  algo_sentiment_daily.csv  (date-level sentiment)
         algo_history.csv          (1-min OHLCV, timestamp in ms)
Outputs: sentiment_analysis.png    (price + sentiment_ratio chart)
         prints correlation tables to stdout
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import pearsonr

# ── Load & resample price data ────────────────────────────────────────────────

price_raw = pd.read_csv("algo_history.csv")
price_raw["timestamp"] = pd.to_datetime(price_raw["timestamp"], unit="ms", utc=True)
price_raw = price_raw.set_index("timestamp").sort_index()

# Daily: last close of each UTC day
daily_close = price_raw["close"].resample("1D").last().dropna()
daily_close.index = daily_close.index.date          # plain date for alignment
daily_close.index.name = "date"
daily_returns = daily_close.pct_change()

price_df = pd.DataFrame({
    "close":  daily_close,
    "return": daily_returns,
})

# ── Load sentiment data ───────────────────────────────────────────────────────

sent_df = pd.read_csv("algo_sentiment_daily.csv", parse_dates=["date"])
sent_df["date"] = sent_df["date"].dt.date
sent_df = sent_df.set_index("date").sort_index()

# ── Align: inner join on date ─────────────────────────────────────────────────

merged = price_df.join(sent_df[["sentiment_ratio", "total_posts"]], how="inner")
merged = merged.dropna(subset=["return", "sentiment_ratio", "total_posts"])

print(f"Aligned dataset: {len(merged)} days  "
      f"({merged.index.min()} to {merged.index.max()})\n")

# ── Cross-correlation helper ──────────────────────────────────────────────────

def cross_corr_table(signal: pd.Series, returns: pd.Series,
                     lags: list[int], signal_name: str) -> None:
    """
    For each lag k: correlate signal[t] with returns[t+k].
    Positive lag means the signal leads returns.
    """
    print(f"{'-'*54}")
    print(f"  {signal_name}  ->  next-day returns")
    print(f"{'-'*54}")
    print(f"  {'Lag':>4}  {'Corr':>8}  {'p-value':>10}  {'Sig':>4}")
    print(f"{'-'*54}")

    for lag in lags:
        if lag == 0:
            x = signal.values
            y = returns.values
        else:
            # signal at t predicts returns at t+lag
            x = signal.values[:-lag]
            y = returns.values[lag:]

        # drop any NaN pairs
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 5:
            print(f"  {lag:>4}  {'—':>8}  {'—':>10}  (insufficient data)")
            continue

        corr, pval = pearsonr(x[mask], y[mask])
        sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.10 else ""
        print(f"  {lag:>4}  {corr:>+8.4f}  {pval:>10.4f}  {sig:>4}")

    print(f"{'-'*54}")
    print(f"  n (lag-0) = {np.isfinite(signal.values).sum()}")
    print()

# ── Run tables ────────────────────────────────────────────────────────────────

LAGS = [0, 1, 2, 3, 5, 7]

cross_corr_table(merged["sentiment_ratio"], merged["return"], LAGS,
                 "sentiment_ratio")
cross_corr_table(merged["total_posts"],    merged["return"], LAGS,
                 "total_posts    ")

print("Significance: *** p<0.01  ** p<0.05  * p<0.10\n")

# ── Descriptive stats ─────────────────────────────────────────────────────────

print("Descriptive stats on aligned sample:")
print(merged[["close", "return", "sentiment_ratio", "total_posts"]]
      .describe().round(4).to_string())
print()

# ── Plot ──────────────────────────────────────────────────────────────────────

dates = pd.to_datetime(merged.index)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                gridspec_kw={"height_ratios": [2, 1]})
fig.suptitle("ALGO Daily Close Price vs Reddit Sentiment Ratio", fontsize=13)

# Top: price
ax1.plot(dates, merged["close"].values, color="#1f77b4", linewidth=1.5)
ax1.set_ylabel("Close Price (USDT)", fontsize=10)
ax1.grid(axis="y", linestyle="--", alpha=0.4)
ax1.yaxis.set_major_formatter(plt.FormatStrFormatter("%.4f"))

# Bottom: sentiment_ratio
ax2.fill_between(dates, merged["sentiment_ratio"].values,
                 alpha=0.35, color="#2ca02c")
ax2.plot(dates, merged["sentiment_ratio"].values,
         color="#2ca02c", linewidth=1.2)
ax2.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.7,
            label="neutral (0.5)")
ax2.set_ylabel("Sentiment Ratio\n(positive / total)", fontsize=10)
ax2.set_ylim(0, 1)
ax2.legend(fontsize=8, loc="lower right")
ax2.grid(axis="y", linestyle="--", alpha=0.4)

ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax2.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
fig.autofmt_xdate(rotation=30, ha="right")

plt.tight_layout()
plt.savefig("sentiment_analysis.png", dpi=150, bbox_inches="tight")
print("Plot saved to sentiment_analysis.png")
