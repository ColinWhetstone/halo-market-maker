"""
BTC/ETH Statistical Arbitrage - Live Paper Trader
==================================================
Polls BTC/USD and ETH/USD on Kraken every 10 seconds.
Uses pre-estimated OLS hedge ratio (beta) and intercept (alpha).
Generates +-2-sigma entry / 0-sigma exit signals on rolling z-score.
Logs fills to stat_arb_trades.csv and prints a status line each poll.
Press Ctrl+C to stop and save stat_arb_performance.png.
"""

from __future__ import annotations

import csv
import math
import os
import signal
import time
from collections import deque
from datetime import datetime, timezone

import ccxt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────

ALPHA        = 5.458006   # OLS intercept from historical fit
BETA         = 0.743603   # OLS hedge ratio: log(BTC) = alpha + beta*log(ETH)
ZSCORE_WIN   = 120        # rolling z-score window (bars)
ENTRY_Z      = 2.0        # enter when |z| exceeds this
EXIT_Z       = 0.0        # exit when |z| crosses zero
COST         = 0.001      # 0.1% per trade per leg
POLL_S       = 10         # seconds between polls
TRADES_CSV   = "stat_arb_trades.csv"
CHART_PNG    = "stat_arb_performance.png"
HISTORY_BTC  = "btc_data.csv"
HISTORY_ETH  = "eth_data.csv"

FIELDNAMES = [
    "timestamp", "action", "btc_price", "eth_price",
    "spread", "z_score", "position", "equity", "pnl_pct",
]

# ── State ─────────────────────────────────────────────────────────────────────

spread_buf: deque[float] = deque(maxlen=ZSCORE_WIN)
equity       = 1.0
position     = 0          # +1 long spread, -1 short spread, 0 flat
entry_btc    = 0.0
entry_eth    = 0.0
prev_btc     = 0.0        # last tick's BTC price (for incremental return)
prev_eth     = 0.0        # last tick's ETH price
n_trades     = 0
equity_history: list[float]    = []
time_history:   list[datetime] = []
trade_rows:     list[dict]     = []
stop_event_flag = False

# ── CSV helpers ───────────────────────────────────────────────────────────────

def _ensure_csv() -> None:
    if not os.path.exists(TRADES_CSV) or os.path.getsize(TRADES_CSV) == 0:
        with open(TRADES_CSV, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()


def _append_row(row: dict) -> None:
    with open(TRADES_CSV, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)

# ── Seed z-score buffer from historical CSV data ──────────────────────────────

def _seed_buffer() -> None:
    """Pre-fill spread_buf from historical 1-min data so z-score is valid immediately."""
    if not (os.path.exists(HISTORY_BTC) and os.path.exists(HISTORY_ETH)):
        print("  [seed] Historical CSVs not found - buffer will warm up live.")
        return

    try:
        btc_df = pd.read_csv(HISTORY_BTC)
        eth_df = pd.read_csv(HISTORY_ETH)

        btc_df["dt"] = pd.to_datetime(btc_df["timestamp"], unit="ms", utc=True)
        eth_df["dt"] = pd.to_datetime(eth_df["timestamp"], unit="ms", utc=True)

        btc_min = btc_df.set_index("dt")["mid_price"].resample("1min").last().dropna()
        eth_min = eth_df.set_index("dt")["mid_price"].resample("1min").last().dropna()

        prices = pd.DataFrame({"BTC": btc_min, "ETH": eth_min}).dropna()

        spreads = np.log(prices["BTC"].values) - BETA * np.log(prices["ETH"].values) - ALPHA

        # Take last ZSCORE_WIN values to seed the buffer
        seed_vals = spreads[-ZSCORE_WIN:]
        for v in seed_vals:
            spread_buf.append(float(v))

        print(f"  [seed] Seeded z-score buffer with {len(seed_vals)} bars "
              f"from {len(prices):,} historical rows.")
    except Exception as exc:
        print(f"  [seed] Warning: could not seed buffer: {exc!r}")


# ── Chart on exit ─────────────────────────────────────────────────────────────

def _save_chart() -> None:
    if len(equity_history) < 2:
        print("Not enough data points to plot.")
        return

    eq   = np.array(equity_history)
    times = time_history

    fig, axes = plt.subplots(2, 1, figsize=(13, 8),
                              gridspec_kw={"height_ratios": [3, 2], "hspace": 0.35})
    fig.patch.set_facecolor("#080b14")

    for ax in axes:
        ax.set_facecolor("#101526")
        ax.tick_params(colors="#3a4060", labelsize=8)
        ax.spines["bottom"].set_color("#1c2238")
        ax.spines["left"].set_color("#1c2238")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color=(1, 1, 1, 0.06), linewidth=0.5, linestyle="--")
        ax.yaxis.label.set_color("#6b749e")
        ax.xaxis.label.set_color("#6b749e")

    total_ret = eq[-1] - 1.0
    ret_series = pd.Series(eq).pct_change().dropna()
    ann_factor = 252 * 390
    sharpe = (ret_series.mean() / ret_series.std(ddof=1) * math.sqrt(ann_factor)
              if ret_series.std() > 0 else 0.0)

    # Panel 1: equity curve
    ax0 = axes[0]
    ax0.plot(times, eq, color="#4f8ef7", linewidth=1.3)
    ax0.axhline(1.0, color=(1, 1, 1, 0.15), linewidth=0.8, linestyle="--")
    ax0.fill_between(times, eq, 1.0,
                     where=eq >= 1.0, alpha=0.12, color="#00d48c")
    ax0.fill_between(times, eq, 1.0,
                     where=eq < 1.0, alpha=0.12, color="#f04e6a")
    ax0.set_ylabel("Equity (normalised)", fontsize=9)
    ax0.set_title(
        f"BTC/ETH Stat Arb Paper Trader  |  Sharpe {sharpe:.2f}  |  "
        f"Return {total_ret:+.2%}  |  {n_trades} trades",
        color="#d8dff5", fontsize=10, pad=8,
    )

    # Panel 2: drawdown
    rolling_max = pd.Series(eq).cummax()
    drawdown = (pd.Series(eq) - rolling_max) / rolling_max
    ax1 = axes[1]
    ax1.fill_between(times, drawdown.values, 0,
                     alpha=0.5, color="#f04e6a")
    ax1.plot(times, drawdown.values, color="#f04e6a", linewidth=0.8)
    ax1.set_ylabel("Drawdown", fontsize=9)
    ax1.axhline(0, color=(1, 1, 1, 0.15), linewidth=0.6)

    import matplotlib.dates as mdates
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate(rotation=30, ha="right")

    plt.savefig(CHART_PNG, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\nChart saved to {CHART_PNG}")


# ── Main loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    global equity, position, entry_btc, entry_eth, prev_btc, prev_eth, n_trades, stop_event_flag

    print("=" * 60)
    print("  BTC/ETH Stat Arb - Live Paper Trader")
    print("=" * 60)
    print(f"  alpha={ALPHA:.6f}  beta={BETA:.6f}")
    print(f"  Entry threshold : +-{ENTRY_Z} z-score")
    print(f"  Exit threshold  : {EXIT_Z} z-score")
    print(f"  Cost per trade  : {COST:.1%} per leg")
    print(f"  Z-score window  : {ZSCORE_WIN} bars")
    print(f"  Poll interval   : {POLL_S}s")
    print()

    _ensure_csv()
    _seed_buffer()

    exchange = ccxt.kraken({"enableRateLimit": True})

    def _handle_sigint(sig, frame):
        global stop_event_flag
        print("\n\nShutting down...")
        stop_event_flag = True

    signal.signal(signal.SIGINT, _handle_sigint)

    print("\nPolling live prices (Ctrl+C to stop)...\n")

    while not stop_event_flag:
        loop_start = time.monotonic()

        try:
            ob_btc = exchange.fetch_order_book("BTC/USD")
            ob_eth = exchange.fetch_order_book("ETH/USD")

            def _mid(ob):
                bids = ob.get("bids", [])
                asks = ob.get("asks", [])
                if not bids or not asks:
                    return None
                return (float(bids[0][0]) + float(asks[0][0])) / 2.0

            btc_price = _mid(ob_btc)
            eth_price = _mid(ob_eth)

            if btc_price is None or eth_price is None:
                print("  [warn] Could not read mid prices, skipping.")
                time.sleep(POLL_S)
                continue

            # Compute spread and z-score
            spread_val = math.log(btc_price) - BETA * math.log(eth_price) - ALPHA
            spread_buf.append(spread_val)

            now_utc = datetime.now(timezone.utc)
            time_str = now_utc.strftime("%H:%M:%S")

            if len(spread_buf) < 30:
                print(f"[{time_str}] warming up z-score buffer "
                      f"({len(spread_buf)}/{ZSCORE_WIN})...")
                elapsed = time.monotonic() - loop_start
                time.sleep(max(0.0, POLL_S - elapsed))
                continue

            buf_arr   = np.array(spread_buf)
            z_score   = (spread_val - buf_arr.mean()) / (buf_arr.std(ddof=1) + 1e-12)

            prev_position = position

            # Entry / exit logic
            if position == 0:
                if z_score < -ENTRY_Z:
                    position = 1
                elif z_score > ENTRY_Z:
                    position = -1
            elif position == 1 and z_score >= EXIT_Z:
                position = 0
            elif position == -1 and z_score <= EXIT_Z:
                position = 0

            # Transaction costs on position change
            action = "hold"
            if position != prev_position:
                if prev_position == 0:
                    equity *= (1 - COST)
                    n_trades += 1
                    entry_btc = btc_price
                    entry_eth = eth_price
                    action = "enter_long" if position == 1 else "enter_short"
                elif position == 0:
                    equity *= (1 - COST)
                    action = "exit"
                else:
                    equity *= (1 - 2 * COST)
                    n_trades += 2
                    entry_btc = btc_price
                    entry_eth = eth_price
                    action = "flip_long" if position == 1 else "flip_short"

            # Incremental bar return on open position
            if position != 0 and prev_btc > 0 and prev_eth > 0:
                r_btc = math.log(btc_price / prev_btc)
                r_eth = math.log(eth_price / prev_eth)
                bar_return = position * (r_btc - BETA * r_eth)
                equity *= math.exp(bar_return)

            pnl_pct = equity - 1.0
            equity_history.append(equity)
            time_history.append(now_utc)

            # Status line
            pos_str = {1: "+1 (long)", -1: "-1 (short)", 0: " 0 (flat)"}[position]
            print(
                f"[{time_str}]  BTC={btc_price:,.2f}  ETH={eth_price:,.2f}  "
                f"spread={spread_val:+.6f}  z={z_score:+.3f}  "
                f"pos={pos_str}  equity={equity:.5f}  pnl={pnl_pct:+.3%}"
            )

            # CSV row (always log, not just on trades)
            row = {
                "timestamp":  now_utc.strftime("%Y-%m-%d %H:%M:%S"),
                "action":     action,
                "btc_price":  round(btc_price, 2),
                "eth_price":  round(eth_price, 2),
                "spread":     round(spread_val, 8),
                "z_score":    round(z_score, 6),
                "position":   position,
                "equity":     round(equity, 8),
                "pnl_pct":    round(pnl_pct * 100, 4),
            }
            _append_row(row)
            trade_rows.append(row)

            # Update previous prices for next tick's incremental return
            prev_btc = btc_price
            prev_eth = eth_price

        except Exception as exc:
            print(f"  [error] {exc!r} - retrying in {POLL_S}s")

        elapsed = time.monotonic() - loop_start
        remaining = POLL_S - elapsed
        if remaining > 0 and not stop_event_flag:
            time.sleep(remaining)

    # ── Shutdown ──────────────────────────────────────────────────────────────

    print(f"\n  Trades logged : {n_trades}")
    print(f"  Final equity  : {equity:.6f}")
    print(f"  Total PnL     : {equity - 1.0:+.3%}")
    _save_chart()


if __name__ == "__main__":
    main()
