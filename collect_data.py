# Collect live market data from Kraken for multiple pairs simultaneously.
# Each pair runs in its own thread and writes to its own CSV file.
# Press Ctrl+C to stop all collectors cleanly.

import csv
import os
import signal
import threading
import time
from typing import Any

import ccxt

# ── Config ────────────────────────────────────────────────────────────────────

PAIRS: list[tuple[str, str]] = [
    ("ALGO/USD",  "algo_data.csv"),
    ("SOL/USD",   "sol_data.csv"),
    ("ADA/USD",   "ada_data.csv"),
    ("DOT/USD",   "dot_data.csv"),
    ("ETH/USD",   "eth_data.csv"),
    ("BTC/USD",   "btc_data.csv"),
    ("OCEAN/USD", "ocean_data.csv"),
    ("KAVA/USD",  "kava_data.csv"),
    ("MINA/USD",  "mina_data.csv"),
]

POLL_INTERVAL_S  = 10   # seconds between successful polls
RETRY_INTERVAL_S = 30   # seconds to wait after an error

FIELDNAMES = [
    "timestamp",
    "best_bid",
    "best_ask",
    "bid_volume",
    "ask_volume",
    "mid_price",
    "spread",
    "last_trade_price",
    "last_trade_volume",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def _ensure_csv_has_header(path: str) -> None:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()


def _safe_first_level(levels: list[list[Any]]) -> tuple[float, float] | None:
    if not levels:
        return None
    row = levels[0]
    if len(row) < 2:
        return None
    return float(row[0]), float(row[1])


# ── Per-pair collector ────────────────────────────────────────────────────────

def _collect(symbol: str, csv_path: str, stop_event: threading.Event) -> None:
    """
    Runs forever (until stop_event is set) polling Kraken for `symbol`
    and appending one row per poll to `csv_path`.
    Each pair gets its own ccxt exchange instance to avoid shared-state issues.
    """
    tag = f"[{symbol}]"
    exchange = ccxt.kraken({"enableRateLimit": True})
    _ensure_csv_has_header(csv_path)
    print(f"{tag} collector started -> {csv_path}")

    while not stop_event.is_set():
        loop_start = time.monotonic()
        try:
            order_book = exchange.fetch_order_book(symbol)
            trades     = exchange.fetch_trades(symbol, limit=1)

            best_bid = _safe_first_level(order_book.get("bids", []))
            best_ask = _safe_first_level(order_book.get("asks", []))

            last_trade_price:  float | None = None
            last_trade_volume: float | None = None
            if trades:
                t = trades[-1]
                last_trade_price  = float(t["price"])  if t.get("price")  is not None else None
                last_trade_volume = float(t["amount"]) if t.get("amount") is not None else None

            timestamp_ms = exchange.milliseconds()

            if best_bid is None or best_ask is None:
                row: dict = {
                    "timestamp": timestamp_ms,
                    "best_bid": None, "best_ask": None,
                    "bid_volume": None, "ask_volume": None,
                    "mid_price": None, "spread": None,
                    "last_trade_price": last_trade_price,
                    "last_trade_volume": last_trade_volume,
                }
            else:
                bid_price, bid_vol = best_bid
                ask_price, ask_vol = best_ask
                row = {
                    "timestamp":        timestamp_ms,
                    "best_bid":         bid_price,
                    "best_ask":         ask_price,
                    "bid_volume":       bid_vol,
                    "ask_volume":       ask_vol,
                    "mid_price":        (bid_price + ask_price) / 2.0,
                    "spread":           ask_price - bid_price,
                    "last_trade_price": last_trade_price,
                    "last_trade_volume":last_trade_volume,
                }

            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)

            # Sleep for the remainder of the poll interval
            elapsed = time.monotonic() - loop_start
            stop_event.wait(max(0.0, POLL_INTERVAL_S - elapsed))

        except Exception as exc:
            print(f"{tag} error: {exc!r} — retrying in {RETRY_INTERVAL_S}s")
            stop_event.wait(RETRY_INTERVAL_S)

    print(f"{tag} collector stopped.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    stop_event = threading.Event()

    # Ctrl+C sets the stop event so all threads exit cleanly
    def _handle_sigint(sig, frame):
        print("\nShutting down all collectors...")
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_sigint)

    threads = [
        threading.Thread(
            target=_collect,
            args=(symbol, csv_path, stop_event),
            name=symbol,
            daemon=True,
        )
        for symbol, csv_path in PAIRS
    ]

    print(f"Starting {len(threads)} collectors (Ctrl+C to stop)...")
    for t in threads:
        t.start()

    # Block main thread until stop_event is set by Ctrl+C
    stop_event.wait()

    # Give threads a moment to finish their current write
    for t in threads:
        t.join(timeout=5)

    print("All collectors stopped.")


if __name__ == "__main__":
    main()
