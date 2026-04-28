"""
deribit_surface.py — BTC options chain from Deribit public API.

Pulls instruments, fetches mark IV + greeks via the ticker endpoint,
filters to the nearest 6 expiries, and prints a combined call/put table
per expiry (strike | call IV | call delta | put IV | put delta).

No authentication required.

Usage:
    python deribit_surface.py
    python deribit_surface.py --expiries 4      # show 4 nearest expiries
    python deribit_surface.py --workers 20      # parallel fetch concurrency
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Optional

import requests

BASE = "https://www.deribit.com/api/v2/public"
SESSION = requests.Session()
SESSION.headers["User-Agent"] = "deribit-surface/1.0"


# ── API helpers ───────────────────────────────────────────────────────────────

def _get(endpoint: str, params: dict) -> dict:
    url = f"{BASE}/{endpoint}"
    resp = SESSION.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise RuntimeError(f"Deribit error: {data['error']}")
    return data["result"]


def fetch_instruments(currency: str = "BTC") -> list[dict]:
    return _get("get_instruments", {"currency": currency, "kind": "option"})


def fetch_ticker(instrument_name: str) -> Optional[dict]:
    try:
        return _get("ticker", {"instrument_name": instrument_name})
    except Exception:
        return None


# ── Data collection ───────────────────────────────────────────────────────────

def collect_surface(n_expiries: int = 6, max_workers: int = 30) -> dict:
    """
    Returns a nested dict:
        { expiry_str: { strike: { "C": ticker, "P": ticker } } }
    for the nearest n_expiries.
    """
    print("Fetching instrument list...", flush=True)
    instruments = fetch_instruments()

    # Parse and sort unique expiries (instruments carry expiration_timestamp in ms)
    now_ms = datetime.now(tz=timezone.utc).timestamp() * 1000
    # Keep only instruments that haven't expired yet
    live = [i for i in instruments if i["expiration_timestamp"] > now_ms]

    expiry_dates: dict[str, int] = {}   # expiry_str -> timestamp_ms
    for inst in live:
        # instrument_name format: BTC-27DEC24-50000-C
        parts = inst["instrument_name"].split("-")
        expiry_str = parts[1]           # e.g. "27DEC24"
        expiry_dates[expiry_str] = inst["expiration_timestamp"]

    # Select nearest n_expiries
    sorted_expiries = sorted(expiry_dates, key=lambda e: expiry_dates[e])
    chosen = sorted_expiries[:n_expiries]
    chosen_set = set(chosen)

    # Filter instruments to chosen expiries
    target = [i for i in live if i["instrument_name"].split("-")[1] in chosen_set]
    print(f"Fetching tickers for {len(target)} instruments across "
          f"{len(chosen)} expiries...", flush=True)

    # Parallel ticker fetch
    results: dict[str, dict] = {}   # instrument_name -> ticker
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(fetch_ticker, inst["instrument_name"]): inst["instrument_name"]
            for inst in target
        }
        done = 0
        for fut in as_completed(futures):
            name = futures[fut]
            ticker = fut.result()
            if ticker is not None:
                results[name] = ticker
            done += 1
            print(f"\r  {done}/{len(target)} fetched", end="", flush=True)
    print()  # newline after progress

    # Organise into surface dict
    surface: dict[str, dict[float, dict[str, dict]]] = {}
    for name, ticker in results.items():
        parts = name.split("-")          # BTC-27DEC24-50000-C
        expiry_str = parts[1]
        strike = float(parts[2])
        option_type = parts[3]           # "C" or "P"

        surface.setdefault(expiry_str, {})
        surface[expiry_str].setdefault(strike, {})
        surface[expiry_str][strike][option_type] = ticker

    # Preserve sorted expiry order
    ordered: dict[str, dict] = {e: surface[e] for e in chosen if e in surface}
    return ordered


# ── Formatting helpers ────────────────────────────────────────────────────────

def _fmt_iv(val) -> str:
    if val is None or val != val:   # None or NaN
        return "   n/a  "
    return f"{val:7.2f}%"


def _fmt_delta(val) -> str:
    if val is None or val != val:
        return "   n/a  "
    return f"{val:+7.4f}"


def _expiry_label(expiry_str: str) -> str:
    """Convert '27DEC24' → '27-DEC-24'."""
    return f"{expiry_str[:2]}-{expiry_str[2:5]}-{expiry_str[5:]}"


# ── Table printer ─────────────────────────────────────────────────────────────

COL_WIDTHS = {
    "strike":      10,
    "call_iv":      9,
    "call_delta":   9,
    "put_iv":       9,
    "put_delta":    9,
}

HEADER = (
    f"{'Strike':>10}  "
    f"{'Call IV':>9}  "
    f"{'Call Delta':>10}  "
    f"{'Put IV':>9}  "
    f"{'Put Delta':>9}"
)
SEP = "-" * len(HEADER)


def _ticker_iv(ticker: Optional[dict]) -> Optional[float]:
    if ticker is None:
        return None
    iv = ticker.get("mark_iv")
    return float(iv) if iv is not None else None


def _ticker_delta(ticker: Optional[dict]) -> Optional[float]:
    if ticker is None:
        return None
    greeks = ticker.get("greeks")
    if not greeks:
        return None
    d = greeks.get("delta")
    return float(d) if d is not None else None


def print_surface(surface: dict) -> None:
    for expiry_str, strikes_dict in surface.items():
        print()
        label = _expiry_label(expiry_str)
        title = f"  Expiry: {label}  ({len(strikes_dict)} strikes)  "
        print("=" * max(len(title), len(SEP)))
        print(title)
        print("=" * max(len(title), len(SEP)))
        print(HEADER)
        print(SEP)

        atm_marker = ""
        prev_call_delta = None

        for strike in sorted(strikes_dict):
            call = strikes_dict[strike].get("C")
            put  = strikes_dict[strike].get("P")

            c_iv    = _ticker_iv(call)
            c_delta = _ticker_delta(call)
            p_iv    = _ticker_iv(put)
            p_delta = _ticker_delta(put)

            # Mark ATM row where call delta crosses 0.50
            atm = ""
            if c_delta is not None and prev_call_delta is not None:
                if prev_call_delta > 0.50 >= c_delta:
                    atm = " << ATM"
            if c_delta is not None:
                prev_call_delta = c_delta

            row = (
                f"{strike:>10,.0f}  "
                f"{_fmt_iv(c_iv):>9}  "
                f"{_fmt_delta(c_delta):>9}  "
                f"{_fmt_iv(p_iv):>9}  "
                f"{_fmt_delta(p_delta):>9}"
                f"{atm}"
            )
            print(row)

        print(SEP)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print BTC options surface from Deribit public API"
    )
    parser.add_argument(
        "--expiries", type=int, default=6,
        help="Number of nearest expiries to display (default: 6)"
    )
    parser.add_argument(
        "--workers", type=int, default=30,
        help="Parallel fetch concurrency (default: 30)"
    )
    args = parser.parse_args()

    try:
        surface = collect_surface(n_expiries=args.expiries, max_workers=args.workers)
    except requests.RequestException as exc:
        print(f"Network error: {exc}", file=sys.stderr)
        sys.exit(1)

    if not surface:
        print("No data returned — check connectivity or try again.", file=sys.stderr)
        sys.exit(1)

    print_surface(surface)
    total_strikes = sum(len(v) for v in surface.values())
    print(f"\nShowing {len(surface)} expiries, {total_strikes} unique strikes.")


if __name__ == "__main__":
    main()
