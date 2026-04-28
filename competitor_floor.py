"""
Competitor floor and repricing velocity estimation from observed market spreads.

- Competitor floor: rolling minimum of observed spreads over a lookback window.
- Repricing velocity: how often the spread has changed over a recent window.
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd


def compute_competitor_floor(
    spreads: pd.Series,
    window: int = 100,
) -> pd.Series:
    """
    Compute a rolling minimum spread (competitor floor) over the given window.

    Parameters
    ----------
    spreads:
        Series of observed market spreads.
    window:
        Rolling window size for the minimum (default: 100 observations).

    Returns
    -------
    floor_series:
        Series with the rolling minimum spread for each timestamp.
    """
    spreads_numeric = pd.to_numeric(spreads, errors="coerce")
    floor_series = spreads_numeric.rolling(window=window, min_periods=1).min()
    return floor_series


def compute_repricing_velocity(
    spreads: pd.Series,
    window: int = 20,
) -> pd.Series:
    """
    Compute repricing velocity: number of times the spread changed in the last `window` observations.

    For each time t:
    - Look back over the last `window` spreads (including t).
    - Count how many times the spread value differs from the previous one in that slice.

    Parameters
    ----------
    spreads:
        Series of observed market spreads.
    window:
        Lookback window over which to count changes (default: 20 observations).

    Returns
    -------
    velocity_series:
        Series of integers (or floats) representing change counts per window.
    """
    spreads_numeric = pd.to_numeric(spreads, errors="coerce")

    # A change occurs whenever current spread != previous spread.
    changed = spreads_numeric != spreads_numeric.shift(1)

    # Rolling sum of "changed" flags over the chosen window gives a count of repricings.
    velocity_series = changed.rolling(window=window, min_periods=1).sum()
    return velocity_series


if __name__ == "__main__":
    csv_path = "algo_data.csv"
    df = pd.read_csv(csv_path)

    if "spread" not in df.columns:
        raise KeyError(f"'spread' column not found in {csv_path}")

    spreads = df["spread"]

    competitor_floor = compute_competitor_floor(spreads)
    repricing_velocity = compute_repricing_velocity(spreads)

    out = pd.DataFrame(
        {
            "spread": spreads,
            "competitor_floor": competitor_floor,
            "repricing_velocity": repricing_velocity,
        }
    )

    print("Last 10 competitor floor estimates and repricing velocities:")
    print(out.tail(10)[["competitor_floor", "repricing_velocity"]])


