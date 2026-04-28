"""
HALO paper trading simulation — ALGO/USD on Kraken.

Fetches live order book + recent trades every 10 seconds via ccxt.
Uses as_engine.py, regime.py, and competitor_floor.py to compute quotes.
Simulates fills when the market crosses our posted quotes.
No real orders are ever placed.

Usage:
    python paper_trader.py            # default 10-second poll cadence
    python paper_trader.py --poll 5   # poll every 5 seconds

Press Ctrl+C to stop and generate paper_performance.png.
Fill log is written incrementally to paper_trades.csv.
"""

from __future__ import annotations

import argparse
import collections
import csv
import math
import os
import random
import signal
import sys
import time
from datetime import datetime, timezone

import ccxt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from as_engine import get_quotes
from competitor_floor import compute_competitor_floor
from regime import compute_volatility_regime

# ── Strategy parameters (mirror strategy.py) ─────────────────────────────────

KAPPA            = 15_000.0
TIME_HORIZON     = 1.0
INVENTORY_CAP    = 50.0
FILL_QTY         = 10.0
MIN_SPREAD_BPS   = 0.002        # 20 bps total spread floor
MAX_SPREAD_BPS   = 0.005        # 50 bps total spread cap
EMERGENCY_THRESH = 35.0
EMERGENCY_MULT   = 5.0
FLOOR_WINDOW     = 100          # lookback for competitor floor
REGIME_WINDOW    = 500          # mid-price history for autocorr regime
VOL_SPAN         = 20           # EWM span for realised volatility
GAMMA_MAP        = {"low": 0.6, "normal": 1.0, "high": 2.5}

# ── I/O ───────────────────────────────────────────────────────────────────────

SYMBOL        = "SOL/USD"
EXCHANGE_ID   = "kraken"
TRADES_CSV    = "paper_trades.csv"
PERF_PNG      = "paper_performance.png"
TRADES_FIELDS = [
    "timestamp", "side", "price", "quantity",
    "spread_captured", "inventory_after", "equity_after",
]

# ── Rolling state ─────────────────────────────────────────────────────────────

mid_history    = collections.deque(maxlen=REGIME_WINDOW + 10)
spread_history = collections.deque(maxlen=FLOOR_WINDOW + 10)

# Performance series (appended every poll)
ts_series:      list[datetime] = []
equity_series:  list[float]    = []
inv_series:     list[float]    = []
sc_pnl_series:  list[float]    = []

# Account state
inventory   = 0.0
cash        = 50_000.0
n_fills     = 0
sc_pnl      = 0.0      # spread-capture PnL
last_mid    = float("nan")

# Fill cooldown — prevent two fills in the same poll tick
last_fill_ts: float = 0.0
FILL_COOLDOWN_S = 20.0   # at least one poll gap between fills

# ── CSV log ───────────────────────────────────────────────────────────────────

def _init_csv() -> None:
    if not os.path.exists(TRADES_CSV) or os.path.getsize(TRADES_CSV) == 0:
        with open(TRADES_CSV, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=TRADES_FIELDS).writeheader()


def _log_fill(ts: datetime, side: str, price: float, qty: float,
              spread: float, inv_after: float, equity_after: float) -> None:
    row = {
        "timestamp":      ts.isoformat(),
        "side":           side,
        "price":          round(price, 8),
        "quantity":       qty,
        "spread_captured": round(spread, 8),
        "inventory_after": round(inv_after, 4),
        "equity_after":   round(equity_after, 4),
    }
    with open(TRADES_CSV, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=TRADES_FIELDS).writerow(row)


# ── Quote computation ─────────────────────────────────────────────────────────

def _compute_quotes(mid: float, spread_obs: float) -> tuple[float, float, str, float]:
    """
    Return (bid, ask, regime, quoted_spread) for the current market state.
    Uses the full rolling history for regime and volatility.
    """
    # Regime from rolling mid-price history
    if len(mid_history) >= 20:
        mid_series   = pd.Series(list(mid_history))
        regimes, _, _ = compute_volatility_regime(mid_series)
        regime        = str(regimes.iloc[-1])
    else:
        regime = "normal"

    # Realised volatility (EWM std of pct returns)
    if len(mid_history) >= 5:
        mid_arr = np.array(list(mid_history), dtype=float)
        returns = np.diff(mid_arr) / mid_arr[:-1]
        alpha   = 2.0 / (VOL_SPAN + 1)
        vol     = float(pd.Series(returns).ewm(span=VOL_SPAN, adjust=False).std().iloc[-1])
    else:
        vol = 0.001   # fallback during cold start

    # Gamma: regime-adaptive + emergency widening on large inventory
    gamma = GAMMA_MAP.get(regime, 1.0)
    if abs(inventory) > EMERGENCY_THRESH:
        gamma *= EMERGENCY_MULT

    # Raw AS quotes
    bid_raw, ask_raw = get_quotes(
        mid_price=mid, inventory=inventory, volatility=vol,
        gamma=gamma, time_horizon=TIME_HORIZON, kappa=KAPPA,
    )

    # Clamp spread to [10 bps, 50 bps]
    r     = 0.5 * (bid_raw + ask_raw)
    s     = ask_raw - bid_raw
    min_s = MIN_SPREAD_BPS * mid
    max_s = MAX_SPREAD_BPS * mid
    s     = max(min_s, min(max_s, s))

    # Raise to competitor floor if available
    if len(spread_history) >= 10:
        floor_series = compute_competitor_floor(
            pd.Series(list(spread_history)), window=FLOOR_WINDOW
        )
        cf = float(floor_series.iloc[-1])
        if math.isfinite(cf) and cf > min_s:
            s = max(s, min(cf, max_s))

    bid = r - 0.5 * s
    ask = r + 0.5 * s
    return bid, ask, regime, s


# ── Fill simulation ───────────────────────────────────────────────────────────

def _try_fill(now: datetime, mid: float,
              bid: float, ask: float, spread: float,
              best_bid_mkt: float, best_ask_mkt: float) -> None:
    """
    Simulate a fill when our quote is competitive.
      - Buy fill : our bid >= market best_bid  (we're on top of the book)
      - Sell fill: our ask <= market best_ask  (we're on top of the book)
    15% fill probability per tick when condition is met.
    Only one fill per cooldown window.
    """
    global inventory, cash, n_fills, sc_pnl, last_fill_ts

    now_ts = now.timestamp()
    if (now_ts - last_fill_ts) < FILL_COOLDOWN_S:
        return

    can_buy  = inventory <  INVENTORY_CAP
    can_sell = inventory > -INVENTORY_CAP

    side  = None
    price = float("nan")

    if can_buy and bid >= best_bid_mkt:
        side  = "buy"
        price = bid
    elif can_sell and ask <= best_ask_mkt:
        side  = "sell"
        price = ask

    if side is not None and random.random() < 0.05:
        return

    if side is None:
        return

    # Toxic flow filter: if the recent mid-price trend runs against the fill
    # direction, skip with 70% probability to avoid being picked off.
    if len(mid_history) >= 10:
        recent = np.array(list(mid_history)[-10:])
        slope  = np.polyfit(np.arange(10), recent, 1)[0]
        if (side == "buy"  and slope < 0 and random.random() < 0.70) or \
           (side == "sell" and slope > 0 and random.random() < 0.70):
            return

    half_spread = 0.5 * spread

    if side == "buy":
        inventory += FILL_QTY
        cash      -= price * FILL_QTY
        sc_pnl    += half_spread * FILL_QTY
    else:
        inventory -= FILL_QTY
        cash      += price * FILL_QTY
        sc_pnl    += half_spread * FILL_QTY

    n_fills     += 1
    last_fill_ts = now_ts
    equity_now   = cash + inventory * mid
    inv_mtm      = equity_now - 500.0 - sc_pnl

    _log_fill(now, side, price, FILL_QTY, half_spread * FILL_QTY,
              inventory, equity_now)

    print(f"  *** FILL: {side.upper()} {FILL_QTY:.0f} @ {price:.6f} | "
          f"inv={inventory:+.0f}  equity={equity_now:.4f}")


# ── Terminal status line ──────────────────────────────────────────────────────

def _print_status(now: datetime, mid: float, bid: float, ask: float,
                  regime: str) -> None:
    equity    = cash + inventory * mid
    inv_mtm   = equity - 500.0 - sc_pnl
    warmup_ok = len(mid_history) >= REGIME_WINDOW
    regime_str = regime if warmup_ok else f"{regime}*"

    print(
        f"[{now.strftime('%H:%M:%S')}]  "
        f"mid={mid:.6f}  "
        f"bid={bid:.6f}  ask={ask:.6f}  "
        f"regime={regime_str:<8}  "
        f"inv={inventory:+7.1f}  "
        f"equity={equity:8.4f}  "
        f"sc_pnl={sc_pnl:+8.4f}  "
        f"mtm={inv_mtm:+8.4f}  "
        f"fills={n_fills}"
    )
    if not warmup_ok:
        remaining = REGIME_WINDOW - len(mid_history)
        print(f"  [warmup: {remaining} more ticks needed for regime signal]")


# ── Performance chart ─────────────────────────────────────────────────────────

def _save_chart() -> None:
    if len(ts_series) < 2:
        print("Not enough data to plot.")
        return

    dates   = ts_series
    equity  = np.array(equity_series)
    inv     = np.array(inv_series)
    sc_cum  = np.array(sc_pnl_series)

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True,
                              gridspec_kw={"hspace": 0.35})
    fig.suptitle("HALO Paper Trading — SOL/USD", fontsize=13)

    # Panel 1: equity curve
    axes[0].plot(dates, equity, color="#1f77b4", linewidth=1.5)
    axes[0].axhline(500.0, color="grey", linestyle="--", linewidth=0.8,
                    label="Starting capital ($500)")
    axes[0].set_ylabel("Equity ($)", fontsize=10)
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)

    # Panel 2: inventory
    axes[1].fill_between(dates, inv, alpha=0.35,
                         color="#2ca02c", where=np.array(inv) >= 0,
                         label="Long")
    axes[1].fill_between(dates, inv, alpha=0.35,
                         color="#d62728", where=np.array(inv) < 0,
                         label="Short")
    axes[1].plot(dates, inv, color="#333333", linewidth=0.8)
    axes[1].axhline(0, color="grey", linestyle="--", linewidth=0.6)
    axes[1].axhline( INVENTORY_CAP, color="red", linestyle=":", linewidth=0.8,
                     alpha=0.6, label="Inv cap")
    axes[1].axhline(-INVENTORY_CAP, color="red", linestyle=":", linewidth=0.8,
                     alpha=0.6)
    axes[1].set_ylabel("Inventory (units)", fontsize=10)
    axes[1].legend(fontsize=8, loc="upper left")
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)

    # Panel 3: cumulative spread-capture PnL
    axes[2].fill_between(dates, sc_cum, alpha=0.3, color="#ff7f0e")
    axes[2].plot(dates, sc_cum, color="#ff7f0e", linewidth=1.4,
                 label="Cumulative spread capture")
    axes[2].set_ylabel("Spread Capture PnL ($)", fontsize=10)
    axes[2].legend(fontsize=8)
    axes[2].grid(axis="y", linestyle="--", alpha=0.4)

    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[2].xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate(rotation=30, ha="right")

    plt.savefig(PERF_PNG, dpi=150, bbox_inches="tight")
    print(f"Chart saved to {PERF_PNG}")


# ── Graceful shutdown ─────────────────────────────────────────────────────────

_shutdown = False

def _handle_sigint(sig, frame):
    global _shutdown
    print("\nCtrl+C received — shutting down...")
    _shutdown = True


# ── Main loop ─────────────────────────────────────────────────────────────────

def main(poll_seconds: int = 10) -> None:
    global inventory, cash, last_mid

    signal.signal(signal.SIGINT, _handle_sigint)
    _init_csv()

    exchange = ccxt.kraken({"enableRateLimit": True})

    print("=" * 72)
    print("  HALO Paper Trader — SOL/USD on Kraken")
    print(f"  Poll cadence : {poll_seconds}s  |  Fill qty : {FILL_QTY:.0f} units")
    print(f"  Inventory cap: +/-{INVENTORY_CAP:.0f}  |  Starting cash: $500")
    print(f"  Regime warmup: {REGIME_WINDOW} ticks (~{REGIME_WINDOW*poll_seconds//60} min)")
    print(f"  Fill log     : {TRADES_CSV}")
    print("  *** NO REAL ORDERS ARE PLACED ***")
    print("  Press Ctrl+C to stop and save chart.")
    print("=" * 72)

    step = 0

    while not _shutdown:
        loop_start = time.monotonic()
        now        = datetime.now(tz=timezone.utc)

        try:
            ob     = exchange.fetch_order_book(SYMBOL)
            trades = exchange.fetch_trades(SYMBOL, limit=1)
        except Exception as exc:
            print(f"[{now.strftime('%H:%M:%S')}] fetch error: {exc!r} — retrying in {poll_seconds}s")
            time.sleep(poll_seconds)
            continue

        # ── Parse market data ─────────────────────────────────────────────────

        bids = ob.get("bids", [])
        asks = ob.get("asks", [])
        if not bids or not asks:
            print(f"[{now.strftime('%H:%M:%S')}] empty order book — skipping")
            time.sleep(poll_seconds)
            continue

        best_bid_mkt  = float(bids[0][0])
        best_ask_mkt  = float(asks[0][0])
        mid           = (best_bid_mkt + best_ask_mkt) / 2.0
        spread_obs    = best_ask_mkt - best_bid_mkt

        # Update rolling history
        mid_history.append(mid)
        spread_history.append(spread_obs)
        last_mid = mid

        # ── Compute quotes ────────────────────────────────────────────────────

        bid, ask, regime, quoted_spread = _compute_quotes(mid, spread_obs)

        # ── Simulate fills ────────────────────────────────────────────────────

        _try_fill(now, mid, bid, ask, quoted_spread,
                  best_bid_mkt, best_ask_mkt)

        # ── Record performance snapshot ───────────────────────────────────────

        equity = cash + inventory * mid
        inv_mtm_pnl = equity - 500.0 - sc_pnl
        ts_series.append(now)
        equity_series.append(equity)
        inv_series.append(inventory)
        sc_pnl_series.append(sc_pnl)

        # ── Print status ──────────────────────────────────────────────────────

        _print_status(now, mid, bid, ask, regime)
        step += 1

        # ── Sleep for remainder of poll interval ──────────────────────────────

        elapsed = time.monotonic() - loop_start
        sleep_s = max(0.0, poll_seconds - elapsed)
        if not _shutdown:
            time.sleep(sleep_s)

    # ── Shutdown ──────────────────────────────────────────────────────────────

    final_equity = cash + inventory * last_mid if math.isfinite(last_mid) else cash
    inv_mtm_final = final_equity - 500.0 - sc_pnl

    print("\n" + "=" * 72)
    print("  Final Performance Summary")
    print("=" * 72)
    print(f"  Ticks recorded  : {len(ts_series)}")
    print(f"  Total fills     : {n_fills}")
    print(f"  Final inventory : {inventory:+.1f} units")
    print(f"  Final equity    : ${final_equity:.4f}  (started $500)")
    print(f"  Total PnL       : ${final_equity - 500.0:+.4f}")
    print(f"  Spread capture  : ${sc_pnl:+.4f}")
    print(f"  Inventory MtM   : ${inv_mtm_final:+.4f}")
    print("=" * 72)

    _save_chart()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HALO paper trader — ALGO/USD on Kraken (no real orders)"
    )
    parser.add_argument(
        "--poll", type=int, default=10,
        help="Seconds between each market data fetch (default: 10)"
    )
    args = parser.parse_args()
    main(poll_seconds=args.poll)
