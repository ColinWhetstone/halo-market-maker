"""
Volatility regime classification using rolling return autocorrelation.

For each timestamp the lag-1 autocorrelation of returns is computed over a
rolling window of 500 observations. The sign and magnitude of this statistic
distinguishes mean-reverting from trending markets without the upward bias
that afflicts the R/S Hurst estimator on short financial series.

Classification rules (applied to autocorr only):
    high   (trending)       autocorr >  0.05
    low    (mean-reverting)  autocorr < -0.05
    normal                   otherwise (including NaN during warmup)
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


# ── Autocorrelation helper ────────────────────────────────────────────────────

def _autocorr_lag1(window: np.ndarray) -> float:
    """Pearson lag-1 autocorrelation of a raw return window."""
    arr = window[np.isfinite(window)]
    if len(arr) < 10:
        return float("nan")
    x, y = arr[:-1], arr[1:]
    std_x = np.std(x, ddof=0)
    std_y = np.std(y, ddof=0)
    if std_x == 0.0 or std_y == 0.0:
        return float("nan")
    return float(np.mean((x - x.mean()) * (y - y.mean())) / (std_x * std_y))


def compute_return_autocorr(
    mid_prices: pd.Series,
    window: int = 500,
) -> pd.Series:
    """Rolling lag-1 autocorrelation of percentage returns over `window` bars."""
    returns = pd.to_numeric(mid_prices, errors="coerce").pct_change()
    return returns.rolling(window=window, min_periods=window).apply(
        _autocorr_lag1, raw=True
    )


# ── Public API ────────────────────────────────────────────────────────────────

def compute_volatility_regime(
    mid_prices: pd.Series,
    short_window: int = 20,
    long_window: int = 200,
    high_threshold: float = 1.5,
    low_threshold: float = 0.7,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Compute regime labels, volatility ratio, and lag-1 return autocorrelation.

    The volatility ratio (short_vol / long_vol) is retained for reference but
    regime classification is driven entirely by autocorrelation:

        high   — autocorr >  0.05  (trending / momentum)
        low    — autocorr < -0.05  (mean-reverting)
        normal — otherwise, including the warmup window where autocorr is NaN

    Returns
    -------
    regime_labels : pd.Series of str  ('high', 'normal', 'low')
    volatility_ratio : pd.Series of float  (short_vol / long_vol, for reference)
    autocorr : pd.Series of float  (rolling lag-1 autocorrelation of returns)
    """
    prices = pd.to_numeric(mid_prices, errors="coerce")
    returns = prices.pct_change()

    short_vol = returns.ewm(span=short_window, adjust=False).std()
    long_vol  = returns.ewm(span=long_window,  adjust=False).std()
    vol_ratio = short_vol / long_vol

    autocorr = compute_return_autocorr(prices, window=500)

    regime_labels = pd.Series("normal", index=prices.index, dtype="object")
    regime_labels[autocorr >  0.05] = "high"
    regime_labels[autocorr < -0.05] = "low"

    return regime_labels, vol_ratio, autocorr


if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "algo_data.csv"
    df = pd.read_csv(csv_path)
    col = "mid_price" if "mid_price" in df.columns else "close"
    regimes, vol_ratio, autocorr = compute_volatility_regime(df[col])
    out = pd.DataFrame({"vol_ratio": vol_ratio, "autocorr": autocorr, "regime": regimes})
    print(out.tail(10).to_string())
    print("\nRegime distribution:")
    print(regimes.value_counts(normalize=True).mul(100).round(1).to_string())
