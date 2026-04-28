import csv
import os
import time
from datetime import datetime, timedelta, timezone

import ccxt


def _ensure_csv_has_header(path: str, fieldnames: list[str]) -> None:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()


def download_history() -> None:
    # Use KuCoin as the data source for historical OHLCV.
    exchange = ccxt.kucoin({"enableRateLimit": True})

    symbol = "ALGO/USDT"
    timeframe = "1m"
    csv_path = "algo_history.csv"
    fieldnames = ["timestamp", "open", "high", "low", "close", "volume"]
    # Start from a clean file each run.
    if os.path.exists(csv_path):
        os.remove(csv_path)
    _ensure_csv_has_header(csv_path, fieldnames)

    # 90 days back from now, in milliseconds.
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    ninety_days_ms = int(timedelta(days=90).total_seconds() * 1000)
    since = now_ms - ninety_days_ms

    # Kraken typically supports up to ~720 OHLCV candles per request; use that as chunk size.
    limit = 720

    print(f"Downloading {symbol} {timeframe} OHLCV from Kraken into {csv_path}...")

    total_rows = 0
    chunk_count = 0

    while since < now_ms:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        except Exception as e:
            print(f"[download_history] error fetching OHLCV: {e!r}")
            # Back off briefly and retry the same 'since'.
            time.sleep(10)
            continue

        rows_to_write: list[dict] = []
        for candle in ohlcv or []:
            ts, o, h, l, c, v = candle  # [timestamp, open, high, low, close, volume]
            rows_to_write.append(
                {
                    "timestamp": ts,
                    "open": o,
                    "high": h,
                    "low": l,
                    "close": c,
                    "volume": v,
                }
            )

        if not rows_to_write:
            print("No more OHLCV data returned; stopping.")
            break

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerows(rows_to_write)

        chunk_count += 1
        total_rows += len(rows_to_write)
        last_ts = rows_to_write[-1]["timestamp"]
        print(f"Wrote {len(rows_to_write)} rows up to {last_ts} (ms since epoch).")

        if chunk_count % 10 == 0:
            print(f"Progress: {total_rows} rows downloaded so far (chunk {chunk_count}).")

        # Advance 'since' to just after the last returned candle to avoid duplicates.
        # 60_000 ms = 1 minute
        since = last_ts + 60_000

        # Be nice to the API.
        time.sleep(1)


if __name__ == "__main__":
    download_history()

