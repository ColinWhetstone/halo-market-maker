"""
Collects ALGO sentiment from two Reddit sources:
  1. r/algorand      — latest posts  (source: 'reddit_algo')
  2. r/CryptoCurrency — ALGO mentions (source: 'reddit_cc')

Outputs:
  algo_sentiment.csv       — raw posts: timestamp, source, title, sentiment_score
  algo_sentiment_daily.csv — daily aggregates: date, total_posts, avg_sentiment_score,
                             positive_count, negative_count, neutral_count, sentiment_ratio
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

# ── Config ────────────────────────────────────────────────────────────────────

REDDIT_USER_AGENT  = "algo-sentiment-collector/1.0 (research bot)"
REDDIT_ALGO_URL    = "https://www.reddit.com/r/algorand/new.json"
REDDIT_CC_URL      = "https://www.reddit.com/r/CryptoCurrency/search.json"
OUTPUT_RAW         = "algo_sentiment.csv"
OUTPUT_DAILY       = "algo_sentiment_daily.csv"
LOOKBACK_DAYS      = 90

# ── Reddit helpers ────────────────────────────────────────────────────────────

def _score_to_sentiment(score: int) -> int:
    if score > 0:
        return 1
    if score < 0:
        return -1
    return 0


def _fetch_reddit_listing(url: str, base_params: dict, source_label: str) -> pd.DataFrame:
    """
    Page through a Reddit listing endpoint until we hit the 90-day cutoff
    or run out of pages. Returns a DataFrame of raw posts.
    """
    headers = {"User-Agent": REDDIT_USER_AGENT}
    cutoff  = datetime.now(tz=timezone.utc) - timedelta(days=LOOKBACK_DAYS)
    rows: list[dict] = []
    after: str | None = None

    while True:
        params = {**base_params, "limit": 100}
        if after:
            params["after"] = after

        try:
            resp = requests.get(url, params=params, headers=headers, timeout=15)
            resp.raise_for_status()
        except requests.HTTPError as exc:
            print(f"  [{source_label}] HTTP error: {exc}")
            break
        except requests.RequestException as exc:
            print(f"  [{source_label}] Request failed: {exc}")
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

        time.sleep(1.0)   # respect Reddit rate limits

    return pd.DataFrame(rows)


def fetch_reddit_algorand() -> pd.DataFrame:
    return _fetch_reddit_listing(
        REDDIT_ALGO_URL,
        base_params={},
        source_label="reddit_algo",
    )


def fetch_reddit_cc() -> pd.DataFrame:
    return _fetch_reddit_listing(
        REDDIT_CC_URL,
        base_params={"q": "ALGO", "sort": "new"},
        source_label="reddit_cc",
    )


# ── Daily aggregation ─────────────────────────────────────────────────────────

def build_daily(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["timestamp"], utc=True).dt.date

    daily = (
        df.groupby("date")
        .agg(
            total_posts      =("sentiment_score", "count"),
            avg_sentiment_score=("sentiment_score", "mean"),
            positive_count   =("sentiment_score", lambda s: (s == 1).sum()),
            negative_count   =("sentiment_score", lambda s: (s == -1).sum()),
            neutral_count    =("sentiment_score", lambda s: (s == 0).sum()),
        )
        .reset_index()
    )
    daily["sentiment_ratio"] = daily["positive_count"] / daily["total_posts"]
    daily["avg_sentiment_score"] = daily["avg_sentiment_score"].round(4)
    daily["sentiment_ratio"]     = daily["sentiment_ratio"].round(4)
    daily = daily.sort_values("date", ascending=False).reset_index(drop=True)
    return daily


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Collecting ALGO sentiment over the last {LOOKBACK_DAYS} days...\n")

    print("Fetching r/algorand (new posts)...")
    algo_df = fetch_reddit_algorand()
    print(f"  -> {len(algo_df)} rows from reddit_algo")

    print("Fetching r/CryptoCurrency (ALGO search)...")
    cc_df = fetch_reddit_cc()
    print(f"  -> {len(cc_df)} rows from reddit_cc")

    combined = (
        pd.concat([algo_df, cc_df], ignore_index=True)
        .sort_values("timestamp", ascending=False)
        .reset_index(drop=True)
    )
    combined.to_csv(OUTPUT_RAW, index=False)
    print(f"\nSaved {len(combined)} total rows to {OUTPUT_RAW}")
    print(f"\nRows per source:\n{combined['source'].value_counts().to_string()}")

    print(f"\nBuilding daily aggregates...")
    daily = build_daily(combined)
    daily.to_csv(OUTPUT_DAILY, index=False)
    print(f"Saved {len(daily)} days to {OUTPUT_DAILY}\n")
    print(daily.to_string(index=False))
