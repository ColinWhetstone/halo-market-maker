"""
ALGO sentiment paper trader — runs as a continuous background process.

Every day at 09:00 local time:
  1. Scrapes r/algorand + r/CryptoCurrency for today's ALGO posts
  2. Appends today's aggregate row to algo_sentiment_daily.csv
  3. Evaluates the trading signal:
       LONG if:
         - today's total_posts > 14-day rolling average  (lag-2 signal)
         - sentiment_ratio from 3 days ago > 0.80        (lag-3 signal)
       FLAT otherwise
  4. Fetches live ALGO/USD price from Kraken via ccxt
  5. Simulates entry / exit and logs to sentiment_paper_trades.csv
  6. Prints a daily status report

Between checks: sleeps and prints a live countdown to the next run.

*** NO REAL ORDERS ARE PLACED ***

Usage:
    python sentiment_paper_trader.py            # next 09:00 trigger
    python sentiment_paper_trader.py --now      # run signal check immediately, then wait
    python sentiment_paper_trader.py --hour 14  # trigger at 14:00 instead of 09:00
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone

import ccxt
import pandas as pd
import requests

# ── Config ────────────────────────────────────────────────────────────────────

TRIGGER_HOUR       = 9          # local hour to run daily check
STARTING_CASH      = 500.0
TRADE_QTY_FRAC     = 1.0        # fraction of cash to deploy on entry (1.0 = all in)
SENTIMENT_DAILY    = "algo_sentiment_daily.csv"
RAW_SENTIMENT      = "algo_sentiment.csv"
TRADES_CSV         = "sentiment_paper_trades.csv"
SYMBOL             = "ALGO/USD"

POSTS_MA_WINDOW    = 14
POSTS_LAG          = 2          # total_posts signal leads returns by 2 days
SENT_LAG           = 3          # sentiment_ratio signal leads returns by 3 days
SENT_THRESHOLD     = 0.80

REDDIT_USER_AGENT  = "algo-sentiment-collector/1.0 (research bot)"
REDDIT_ALGO_URL    = "https://www.reddit.com/r/algorand/new.json"
REDDIT_CC_URL      = "https://www.reddit.com/r/CryptoCurrency/search.json"

TRADES_FIELDS = [
    "date", "signal", "action",
    "algo_price", "qty", "cash_after",
    "position_units", "entry_price",
    "equity", "total_return_pct",
    "bh_equity", "bh_return_pct",
    "today_posts", "posts_ma14", "sent_ratio_3d_ago",
]

# ── CSV helpers ───────────────────────────────────────────────────────────────

def _init_trades_csv() -> None:
    if not os.path.exists(TRADES_CSV) or os.path.getsize(TRADES_CSV) == 0:
        with open(TRADES_CSV, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=TRADES_FIELDS).writeheader()


def _append_trade(row: dict) -> None:
    with open(TRADES_CSV, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=TRADES_FIELDS).writerow(row)

# ── Reddit scraping (mirrors sentiment_collector.py) ─────────────────────────

def _score_to_sentiment(score: int) -> int:
    return 1 if score > 0 else (-1 if score < 0 else 0)


def _fetch_reddit_listing(url: str, base_params: dict,
                           source_label: str, lookback_days: int = 3) -> pd.DataFrame:
    """Fetch recent posts; only look back `lookback_days` to keep it fast."""
    headers = {"User-Agent": REDDIT_USER_AGENT}
    cutoff  = datetime.now(tz=timezone.utc) - timedelta(days=lookback_days)
    rows: list[dict] = []
    after: str | None = None

    while True:
        params = {**base_params, "limit": 100}
        if after:
            params["after"] = after
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=15)
            resp.raise_for_status()
        except requests.RequestException as exc:
            print(f"  [{source_label}] fetch error: {exc!r}")
            break

        data     = resp.json()
        children = data.get("data", {}).get("children", [])
        if not children:
            break

        stop = False
        for child in children:
            post    = child.get("data", {})
            created = post.get("created_utc", 0)
            ts      = datetime.fromtimestamp(created, tz=timezone.utc)
            if ts < cutoff:
                stop = True
                break
            rows.append({
                "timestamp":       ts.isoformat(),
                "source":          source_label,
                "title":           post.get("title", "").strip(),
                "sentiment_score": _score_to_sentiment(post.get("score", 0)),
            })

        after = data.get("data", {}).get("after")
        if stop or not after:
            break
        time.sleep(1.0)

    return pd.DataFrame(rows)


def scrape_today() -> dict:
    """
    Scrape today's posts, update algo_sentiment_daily.csv, and return
    today's aggregate stats: {total_posts, sentiment_ratio}.
    """
    today = date.today()
    print(f"  Scraping r/algorand...")
    algo_df = _fetch_reddit_listing(REDDIT_ALGO_URL, {}, "reddit_algo", lookback_days=2)
    print(f"    -> {len(algo_df)} posts")

    print(f"  Scraping r/CryptoCurrency...")
    cc_df = _fetch_reddit_listing(
        REDDIT_CC_URL, {"q": "ALGO", "sort": "new"}, "reddit_cc", lookback_days=2
    )
    print(f"    -> {len(cc_df)} posts")

    combined = pd.concat([algo_df, cc_df], ignore_index=True)

    # Filter to today only
    if not combined.empty:
        combined["date"] = pd.to_datetime(combined["timestamp"], utc=True).dt.date
        today_posts = combined[combined["date"] == today]
    else:
        today_posts = pd.DataFrame()

    total  = len(today_posts)
    pos    = int((today_posts["sentiment_score"] == 1).sum()) if total else 0
    neg    = int((today_posts["sentiment_score"] == -1).sum()) if total else 0
    neu    = int((today_posts["sentiment_score"] == 0).sum()) if total else 0
    avg_s  = round(today_posts["sentiment_score"].mean(), 4) if total else 0.0
    ratio  = round(pos / total, 4) if total > 0 else 0.0

    today_row = {
        "date":                str(today),
        "total_posts":         total,
        "avg_sentiment_score": avg_s,
        "positive_count":      pos,
        "negative_count":      neg,
        "neutral_count":       neu,
        "sentiment_ratio":     ratio,
    }

    # Append raw posts to algo_sentiment.csv
    if not combined.empty and os.path.exists(RAW_SENTIMENT):
        existing_raw = pd.read_csv(RAW_SENTIMENT)
        merged_raw   = pd.concat([existing_raw, combined[["timestamp","source","title","sentiment_score"]]],
                                  ignore_index=True).drop_duplicates(subset=["timestamp","title"])
        merged_raw.to_csv(RAW_SENTIMENT, index=False)

    # Load existing daily CSV, upsert today's row, save
    if os.path.exists(SENTIMENT_DAILY) and os.path.getsize(SENTIMENT_DAILY) > 0:
        daily_df = pd.read_csv(SENTIMENT_DAILY)
        daily_df = daily_df[daily_df["date"] != str(today)]   # remove stale today row
        daily_df = pd.concat(
            [pd.DataFrame([today_row]), daily_df], ignore_index=True
        )
    else:
        daily_df = pd.DataFrame([today_row])

    daily_df.to_csv(SENTIMENT_DAILY, index=False)
    print(f"  Today ({today}): {total} posts, sentiment_ratio={ratio:.4f}")
    return {"total_posts": total, "sentiment_ratio": ratio, "date": today}

# ── Signal computation ────────────────────────────────────────────────────────

def compute_signal() -> dict:
    """
    Load algo_sentiment_daily.csv and compute the long/flat signal.
    Returns a dict with signal details.
    """
    if not os.path.exists(SENTIMENT_DAILY):
        return {"signal": "flat", "reason": "no sentiment data",
                "today_posts": 0, "posts_ma14": None, "sent_3d_ago": None}

    df = pd.read_csv(SENTIMENT_DAILY, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    if len(df) < max(POSTS_LAG, SENT_LAG) + 1:
        return {"signal": "flat", "reason": "insufficient history",
                "today_posts": 0, "posts_ma14": None, "sent_3d_ago": None}

    # Rolling 14-day MA of total_posts
    df["posts_ma14"] = df["total_posts"].rolling(window=POSTS_MA_WINDOW, min_periods=7).mean()

    # Today's row (most recent)
    today_row  = df.iloc[-1]
    today_posts = float(today_row["total_posts"])
    posts_ma    = today_row["posts_ma14"]

    # Sentiment ratio from SENT_LAG days ago
    sent_lag_row = df.iloc[-(SENT_LAG + 1)] if len(df) > SENT_LAG else None
    sent_3d_ago  = float(sent_lag_row["sentiment_ratio"]) if sent_lag_row is not None else None

    posts_cond = (posts_ma is not None) and (today_posts > posts_ma)
    sent_cond  = (sent_3d_ago is not None) and (sent_3d_ago > SENT_THRESHOLD)

    signal = "long" if (posts_cond and sent_cond) else "flat"

    reasons = []
    if not posts_cond:
        reasons.append(f"posts {today_posts:.0f} <= MA {posts_ma:.1f}" if posts_ma else "posts MA unavailable")
    if not sent_cond:
        reasons.append(f"sent_ratio_3d {sent_3d_ago:.4f} <= {SENT_THRESHOLD}" if sent_3d_ago is not None else "sent_3d unavailable")
    reason = "all conditions met" if signal == "long" else " | ".join(reasons)

    return {
        "signal":       signal,
        "reason":       reason,
        "today_posts":  today_posts,
        "posts_ma14":   round(posts_ma, 2) if posts_ma else None,
        "sent_3d_ago":  sent_3d_ago,
    }

# ── Price fetch ───────────────────────────────────────────────────────────────

def fetch_price() -> float:
    exchange = ccxt.kraken({"enableRateLimit": True})
    ob = exchange.fetch_order_book(SYMBOL)
    bid = float(ob["bids"][0][0])
    ask = float(ob["asks"][0][0])
    return (bid + ask) / 2.0

# ── Daily logic ───────────────────────────────────────────────────────────────

class State:
    def __init__(self) -> None:
        self.cash:          float       = STARTING_CASH
        self.position:      float       = 0.0       # units of ALGO held
        self.entry_price:   float       = 0.0
        self.bh_qty:        float       = 0.0       # units for buy-and-hold
        self.bh_entry:      float       = 0.0
        self.last_signal:   str         = "flat"
        self._load_state()

    def _load_state(self) -> None:
        """Resume from last row of trades CSV if it exists."""
        if not os.path.exists(TRADES_CSV) or os.path.getsize(TRADES_CSV) < 10:
            return
        try:
            df = pd.read_csv(TRADES_CSV)
            if df.empty:
                return
            last = df.iloc[-1]
            self.cash        = float(last.get("cash_after",    STARTING_CASH))
            self.position    = float(last.get("position_units",0))
            self.entry_price = float(last.get("entry_price",   0))
            self.last_signal = str(last.get("signal", "flat"))
            print(f"  Resumed from last trade: cash=${self.cash:.4f}  pos={self.position:.2f} ALGO")
        except Exception as exc:
            print(f"  Warning: could not resume state: {exc!r}")

    def equity(self, price: float) -> float:
        return self.cash + self.position * price

    def bh_equity(self, price: float) -> float:
        if self.bh_qty == 0:
            return STARTING_CASH
        return self.bh_qty * price


def run_daily_check(state: State) -> None:
    today = date.today()
    print()
    print("=" * 60)
    print(f"  DAILY CHECK — {today}  {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 60)
    print("  *** NO REAL ORDERS ARE PLACED ***")

    # 1. Scrape
    print("\n[1/4] Scraping Reddit...")
    scrape_today()

    # 2. Signal
    print("\n[2/4] Computing signal...")
    sig = compute_signal()
    signal      = sig["signal"]
    today_posts = sig["today_posts"]
    posts_ma    = sig["posts_ma14"]
    sent_3d     = sig["sent_3d_ago"]
    print(f"  Signal  : {signal.upper()}")
    print(f"  Reason  : {sig['reason']}")
    print(f"  Posts today: {today_posts:.0f}  |  MA14: {posts_ma}  |  Sent(-3d): {sent_3d}")

    # 3. Price
    print("\n[3/4] Fetching ALGO/USD price...")
    try:
        price = fetch_price()
        print(f"  ALGO/USD mid: ${price:.6f}")
    except Exception as exc:
        print(f"  Price fetch failed: {exc!r} — skipping today's update")
        return

    # Initialise buy-and-hold on first ever run
    if state.bh_qty == 0 and state.cash == STARTING_CASH:
        state.bh_qty   = STARTING_CASH / price
        state.bh_entry = price
        print(f"  B&H initialised: {state.bh_qty:.4f} ALGO @ ${price:.6f}")

    # 4. Simulate trade
    print("\n[4/4] Simulating trade...")
    action = "hold"
    qty    = 0.0

    if signal == "long" and state.position == 0:
        # Enter long
        qty              = (state.cash * TRADE_QTY_FRAC) / price
        state.cash      -= qty * price
        state.position   = qty
        state.entry_price= price
        state.last_signal= "long"
        action           = "BUY"
        print(f"  BUY  {qty:.4f} ALGO @ ${price:.6f}  (deployed ${qty*price:.4f})")

    elif signal == "flat" and state.position > 0:
        # Exit long
        qty             = state.position
        state.cash     += qty * price
        state.position  = 0.0
        state.last_signal = "flat"
        action          = "SELL"
        pnl             = (price - state.entry_price) * qty
        state.entry_price = 0.0
        print(f"  SELL {qty:.4f} ALGO @ ${price:.6f}  (PnL: ${pnl:+.4f})")

    else:
        state.last_signal = signal
        print(f"  No action — already {('LONG' if state.position > 0 else 'FLAT')}, signal={signal.upper()}")

    eq          = state.equity(price)
    bh_eq       = state.bh_equity(price)
    total_ret   = (eq - STARTING_CASH) / STARTING_CASH * 100
    bh_ret      = (bh_eq - STARTING_CASH) / STARTING_CASH * 100

    # Log to CSV
    _append_trade({
        "date":                str(today),
        "signal":              signal,
        "action":              action,
        "algo_price":          round(price, 6),
        "qty":                 round(qty, 6),
        "cash_after":          round(state.cash, 4),
        "position_units":      round(state.position, 6),
        "entry_price":         round(state.entry_price, 6),
        "equity":              round(eq, 4),
        "total_return_pct":    round(total_ret, 4),
        "bh_equity":           round(bh_eq, 4),
        "bh_return_pct":       round(bh_ret, 4),
        "today_posts":         int(today_posts),
        "posts_ma14":          round(posts_ma, 2) if posts_ma else "",
        "sent_ratio_3d_ago":   sent_3d if sent_3d is not None else "",
    })

    # Status printout
    print()
    print("-" * 60)
    print(f"  Date          : {today}")
    print(f"  Signal        : {signal.upper()}")
    print(f"  Action        : {action}")
    print(f"  ALGO price    : ${price:.6f}")
    print(f"  Position      : {state.position:.4f} ALGO")
    print(f"  Cash          : ${state.cash:.4f}")
    print(f"  Equity        : ${eq:.4f}")
    print(f"  Total return  : {total_ret:+.2f}%")
    print(f"  Buy-and-hold  : ${bh_eq:.4f}  ({bh_ret:+.2f}%)")
    print("-" * 60)

# ── Scheduler ─────────────────────────────────────────────────────────────────

def seconds_until(hour: int) -> float:
    """Seconds until the next occurrence of `hour:00:00` local time."""
    now  = datetime.now()
    target = now.replace(hour=hour, minute=0, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return (target - now).total_seconds()


def countdown_sleep(total_seconds: float, trigger_hour: int) -> None:
    """Sleep with a live countdown line refreshed every 60 seconds."""
    end_time = datetime.now() + timedelta(seconds=total_seconds)
    print(f"\n  Next check at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

    while True:
        remaining = (end_time - datetime.now()).total_seconds()
        if remaining <= 0:
            break
        hours   = int(remaining // 3600)
        minutes = int((remaining % 3600) // 60)
        sys.stdout.write(
            f"\r  Sleeping... {hours:02d}h {minutes:02d}m until {trigger_hour:02d}:00"
        )
        sys.stdout.flush()
        time.sleep(min(60, remaining))

    print()   # newline after countdown

# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ALGO sentiment paper trader (no real orders)"
    )
    parser.add_argument("--now",  action="store_true",
                        help="Run signal check immediately then resume schedule")
    parser.add_argument("--hour", type=int, default=TRIGGER_HOUR,
                        help=f"Hour to trigger daily check (default: {TRIGGER_HOUR})")
    args = parser.parse_args()

    trigger_hour = args.hour

    print("=" * 60)
    print("  ALGO Sentiment Paper Trader")
    print(f"  Daily trigger : {trigger_hour:02d}:00 local time")
    print(f"  Signal        : posts > MA14 (lag-2) AND sent_ratio > {SENT_THRESHOLD} (lag-3)")
    print(f"  Starting cash : ${STARTING_CASH:.2f}")
    print(f"  Trade log     : {TRADES_CSV}")
    print("  *** NO REAL ORDERS ARE PLACED ***")
    print("=" * 60)

    _init_trades_csv()
    state = State()

    if args.now:
        run_daily_check(state)

    while True:
        wait = seconds_until(trigger_hour)
        countdown_sleep(wait, trigger_hour)
        run_daily_check(state)


if __name__ == "__main__":
    main()
