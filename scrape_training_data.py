"""Scrape historical BTC minute candles from Binance and build training dataset.

Generates a CSV compatible with train_raw_model.py by:
1. Fetching BTC-USDT 1-minute candles from Binance (free, no API key)
2. Simulating Kalshi-style 15-minute binary options at round strike prices
3. Computing all 33 RAW_FEATURES for each synthetic snapshot
4. Labeling outcomes (did BTC close above the strike?)

Usage:
    python3 scrape_training_data.py                  # default: 7 days
    python3 scrape_training_data.py --days 30        # 30 days of data
    python3 scrape_training_data.py --days 90        # 90 days (max ~1000 candles/request)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Coinbase candle endpoint (US-accessible, no auth needed)
COINBASE_CANDLES_URL = "https://api.exchange.coinbase.com/products/BTC-USD/candles"

# Kalshi BTC 15-min market parameters
MARKET_DURATION_SEC = 900  # 15 minutes
STRIKE_ROUND = 100  # Kalshi rounds strikes to nearest $100

# Must match train_raw_model.py RAW_FEATURES
RAW_FEATURES = [
    "seconds_to_close", "entry_bucket",
    "return_1m", "return_3m", "return_5m",
    "volatility_3m", "volatility_5m", "range_5m",
    "abs_return_1m", "trade_count_1m", "trade_count_3m",
    "trade_count_5m", "volume_1m", "volume_3m", "volume_5m",
    "avg_trade_price_1m", "avg_trade_price_3m",
    "momentum_acceleration",
    "flip_count_5m", "return_1m_x_inv_time",
    "return_3m_x_inv_time", "volatility_5m_x_inv_time",
    "bid_ask_spread", "rsi_14", "session",
    "distance_from_strike", "outcome_rate_bucket",
    "return_5m_ratio",
    "was_missing_return_1m", "was_missing_return_3m",
    "was_missing_return_5m", "was_missing_volatility_3m",
    "was_missing_volatility_5m",
]

OUTPUT_COLUMNS = RAW_FEATURES + [
    "price_now", "final_outcome_yes", "close_ts", "market_ticker",
]


# ── Coinbase data fetching ──────────────────────────────────────────────

def fetch_coinbase_candles(start_sec: int, end_sec: int, product: str = "BTC-USD") -> pd.DataFrame:
    """Fetch 1-minute candles from Coinbase in paginated requests.

    Coinbase returns candles as [timestamp, low, high, open, close, volume]
    sorted newest-first, max 300 per request.
    """
    import urllib.request
    import urllib.parse

    all_rows = []
    # Coinbase granularity=60 = 1-minute candles, max 300 per request
    chunk_seconds = 300 * 60  # 300 candles × 60s = 5 hours per request

    current_start = start_sec
    while current_start < end_sec:
        current_end = min(current_start + chunk_seconds, end_sec)
        start_iso = datetime.fromtimestamp(current_start, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        end_iso = datetime.fromtimestamp(current_end, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        params = urllib.parse.urlencode({
            "granularity": 60,
            "start": start_iso,
            "end": end_iso,
        })
        url = f"{COINBASE_CANDLES_URL}?{params}"

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "BTCPred/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())
        except Exception as exc:
            logger.warning("Coinbase request failed: %s — retrying after 2s", exc)
            time.sleep(2)
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "BTCPred/1.0"})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    data = json.loads(resp.read().decode())
            except Exception:
                logger.error("Coinbase request failed twice, stopping pagination")
                break

        if not data:
            current_start = current_end
            continue

        # Coinbase format: [timestamp, low, high, open, close, volume]
        for row in data:
            all_rows.append({
                "ts": int(row[0]),
                "open": float(row[3]),
                "high": float(row[2]),
                "low": float(row[1]),
                "close": float(row[4]),
                "volume": float(row[5]),
            })

        current_start = current_end
        time.sleep(0.3)  # rate limit courtesy

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    logger.info("Fetched %d candles from Coinbase", len(df))
    return df

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["ts"] = df["open_time"] // 1000  # convert ms → seconds
    logger.info("Fetched %d candles from Binance", len(df))
    return df


# ── Feature computation (mirrors app/feature_engineering.py) ───────────

def _closest_bucket(seconds: int) -> int:
    buckets = [60, 120, 180, 300]
    return min(buckets, key=lambda b: abs(b - int(seconds)))


def _compute_rsi(closes: np.ndarray, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes[-(period + 1):])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100.0 - (100.0 / (1.0 + rs)))


def _trading_session(ts_sec: int) -> int:
    hour = datetime.fromtimestamp(int(ts_sec), tz=timezone.utc).hour
    if 0 <= hour < 8:
        return 0
    elif 8 <= hour < 14:
        return 1
    elif 14 <= hour < 21:
        return 2
    else:
        return 0


def _distance_from_strike(btc_price: float, strike: float) -> float:
    if strike == 0:
        return 0.0
    return (btc_price - strike) / strike


def _flip_count(closes: np.ndarray) -> int:
    if len(closes) < 3:
        return 0
    diffs = np.diff(closes)
    signs = np.sign(diffs)
    signs = signs[signs != 0]
    if len(signs) < 2:
        return 0
    return int(np.sum(signs[1:] != signs[:-1]))


def compute_snapshot_features(
    candles: pd.DataFrame,
    snapshot_ts: int,
    close_ts: int,
    strike: float,
    market_ticker: str,
) -> dict | None:
    """Compute all RAW_FEATURES for a single snapshot point."""
    seconds_to_close = close_ts - snapshot_ts
    if seconds_to_close <= 0:
        return None

    # Current price
    eligible = candles[candles["ts"] <= snapshot_ts]
    if eligible.empty:
        return None
    price_now = float(eligible.iloc[-1]["close"])

    # Price at offsets
    def price_at(offset_sec):
        target = snapshot_ts - offset_sec
        rows = candles[candles["ts"] <= target]
        if rows.empty:
            return None
        return float(rows.iloc[-1]["close"])

    price_1m = price_at(60)
    price_3m = price_at(180)
    price_5m = price_at(300)

    was_missing_return_1m = 1 if price_1m is None else 0
    was_missing_return_3m = 1 if price_3m is None else 0
    was_missing_return_5m = 1 if price_5m is None else 0

    price_1m = price_now if price_1m is None else price_1m
    price_3m = price_now if price_3m is None else price_3m
    price_5m = price_now if price_5m is None else price_5m

    return_1m = price_now - price_1m
    return_3m = price_now - price_3m
    return_5m = price_now - price_5m

    # Windows
    c3 = candles[(candles["ts"] > snapshot_ts - 180) & (candles["ts"] <= snapshot_ts)]
    c5 = candles[(candles["ts"] > snapshot_ts - 300) & (candles["ts"] <= snapshot_ts)]

    was_missing_volatility_3m = 1 if len(c3) < 2 else 0
    was_missing_volatility_5m = 1 if len(c5) < 2 else 0

    volatility_3m = float(c3["close"].std(ddof=0)) if len(c3) >= 2 else 0.0
    volatility_5m = float(c5["close"].std(ddof=0)) if len(c5) >= 2 else 0.0

    if len(c5) >= 2:
        range_5m = float(c5["high"].max() - c5["low"].min())
        flip_count_5m = _flip_count(c5["close"].values)
    else:
        range_5m = 0.0
        flip_count_5m = 0

    # Trade/volume proxies from Binance candle data
    t1 = candles[(candles["ts"] > snapshot_ts - 60) & (candles["ts"] <= snapshot_ts)]
    t3 = candles[(candles["ts"] > snapshot_ts - 180) & (candles["ts"] <= snapshot_ts)]
    t5 = candles[(candles["ts"] > snapshot_ts - 300) & (candles["ts"] <= snapshot_ts)]

    trade_count_1m = len(t1)
    trade_count_3m = len(t3)
    trade_count_5m = len(t5)
    volume_1m = float(t1["volume"].sum()) if not t1.empty else 0.0
    volume_3m = float(t3["volume"].sum()) if not t3.empty else 0.0
    volume_5m = float(t5["volume"].sum()) if not t5.empty else 0.0

    avg_trade_price_1m = float((t1["close"] * t1["volume"]).sum() / t1["volume"].sum()) if not t1.empty and t1["volume"].sum() > 0 else price_now
    avg_trade_price_3m = float((t3["close"] * t3["volume"]).sum() / t3["volume"].sum()) if not t3.empty and t3["volume"].sum() > 0 else price_now

    # Derived features
    momentum_acceleration = return_1m - (return_3m / 3.0)
    inv_time = 1.0 / max(seconds_to_close, 1)
    return_1m_x_inv_time = return_1m * inv_time
    return_3m_x_inv_time = return_3m * inv_time
    volatility_5m_x_inv_time = volatility_5m * inv_time

    entry_bucket = _closest_bucket(seconds_to_close)

    # New features
    bid_ask_spread = 0.0  # not available from Binance candle data
    rsi_14 = _compute_rsi(candles[candles["ts"] <= snapshot_ts]["close"].values[-20:])
    session = _trading_session(close_ts)
    distance_from_strike = _distance_from_strike(price_now, strike)
    outcome_rate_bucket = 0.5  # populated later if we have historical data
    return_5m_ratio = (return_1m / return_5m) if abs(return_5m) > 1e-8 else 0.0

    return {
        "seconds_to_close": seconds_to_close,
        "entry_bucket": entry_bucket,
        "return_1m": float(return_1m),
        "return_3m": float(return_3m),
        "return_5m": float(return_5m),
        "volatility_3m": float(volatility_3m),
        "volatility_5m": float(volatility_5m),
        "range_5m": float(range_5m),
        "abs_return_1m": float(abs(return_1m)),
        "trade_count_1m": float(trade_count_1m),
        "trade_count_3m": float(trade_count_3m),
        "trade_count_5m": float(trade_count_5m),
        "volume_1m": float(volume_1m),
        "volume_3m": float(volume_3m),
        "volume_5m": float(volume_5m),
        "avg_trade_price_1m": float(avg_trade_price_1m),
        "avg_trade_price_3m": float(avg_trade_price_3m),
        "momentum_acceleration": float(momentum_acceleration),
        "flip_count_5m": float(flip_count_5m),
        "return_1m_x_inv_time": float(return_1m_x_inv_time),
        "return_3m_x_inv_time": float(return_3m_x_inv_time),
        "volatility_5m_x_inv_time": float(volatility_5m_x_inv_time),
        "bid_ask_spread": float(bid_ask_spread),
        "rsi_14": float(rsi_14),
        "session": int(session),
        "distance_from_strike": float(distance_from_strike),
        "outcome_rate_bucket": float(outcome_rate_bucket),
        "return_5m_ratio": float(return_5m_ratio),
        "was_missing_return_1m": int(was_missing_return_1m),
        "was_missing_return_3m": int(was_missing_return_3m),
        "was_missing_return_5m": int(was_missing_return_5m),
        "was_missing_volatility_3m": int(was_missing_volatility_3m),
        "was_missing_volatility_5m": int(was_missing_volatility_5m),
        "price_now": float(price_now),
        "close_ts": int(close_ts),
        "market_ticker": market_ticker,
    }


# ── Synthetic market generation ────────────────────────────────────────

def generate_markets(candles: pd.DataFrame) -> list[dict]:
    """Create synthetic 15-min Kalshi-style markets aligned to round times.

    Kalshi BTC 15-min markets close at :00, :15, :30, :45 of each hour.
    Strike = BTC price rounded to nearest STRIKE_ROUND at market open.
    """
    if candles.empty:
        return []

    markets = []
    min_ts = int(candles["ts"].min())
    max_ts = int(candles["ts"].max())

    # Align to quarter-hour boundaries
    start_ts = math.ceil(min_ts / 900) * 900

    for close_ts in range(start_ts, max_ts, 900):
        open_ts = close_ts - 900

        # Get BTC price at market open to determine strike
        open_candles = candles[candles["ts"] <= open_ts]
        if open_candles.empty:
            continue
        open_price = float(open_candles.iloc[-1]["close"])

        # Kalshi rounds strikes to nearest $100
        strike = round(open_price / STRIKE_ROUND) * STRIKE_ROUND

        # Get closing price to determine outcome
        close_candles = candles[candles["ts"] <= close_ts]
        if close_candles.empty:
            continue
        close_price = float(close_candles.iloc[-1]["close"])

        final_outcome_yes = 1 if close_price > strike else 0

        # Market ticker format matching Kalshi convention
        close_dt = datetime.fromtimestamp(close_ts, tz=timezone.utc)
        ticker = f"KXBTC15M-{close_dt.strftime('%y%m%d%H%M')}"

        markets.append({
            "close_ts": close_ts,
            "open_ts": open_ts,
            "strike": strike,
            "open_price": open_price,
            "final_outcome_yes": final_outcome_yes,
            "market_ticker": ticker,
        })

    logger.info("Generated %d synthetic markets", len(markets))
    return markets


def build_training_rows(candles: pd.DataFrame, markets: list[dict], snapshots_per_market: int = 4) -> list[dict]:
    """Compute feature snapshots at multiple time points per market."""
    rows = []
    total = len(markets)

    for i, market in enumerate(markets):
        if (i + 1) % 200 == 0:
            logger.info("Processing market %d/%d (%.0f%%)", i + 1, total, (i + 1) / total * 100)

        close_ts = market["close_ts"]
        open_ts = market["open_ts"]
        strike = market["strike"]
        ticker = market["market_ticker"]
        outcome = market["final_outcome_yes"]

        # Take snapshots at: T-60s, T-180s, T-300s, T-600s (entry buckets)
        snapshot_offsets = [60, 180, 300, 600][:snapshots_per_market]

        for offset in snapshot_offsets:
            snapshot_ts = close_ts - offset
            if snapshot_ts < open_ts + 60:
                continue  # too early, not enough candle data

            features = compute_snapshot_features(candles, snapshot_ts, close_ts, strike, ticker)
            if features is None:
                continue

            features["final_outcome_yes"] = outcome
            rows.append(features)

    logger.info("Built %d training rows from %d markets", len(rows), total)
    return rows


# ── Outcome rate bucket computation ────────────────────────────────────

def compute_outcome_rate_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """Fill outcome_rate_bucket using historical YES rate per entry_bucket."""
    if df.empty:
        return df

    rates = df.groupby("entry_bucket")["final_outcome_yes"].mean()
    df["outcome_rate_bucket"] = df["entry_bucket"].map(rates).fillna(0.5)
    return df


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Scrape BTC data and build training dataset")
    parser.add_argument("--days", type=int, default=7, help="Days of history to scrape (default: 7)")
    parser.add_argument("--output", type=str, default="kalshi_btc15m_dataset_scraped.csv",
                        help="Output CSV path")
    parser.add_argument("--snapshots", type=int, default=4,
                        help="Snapshots per market (1-4, default: 4)")
    args = parser.parse_args()

    end_sec = int(time.time())
    start_sec = end_sec - (args.days * 24 * 3600)

    logger.info("Fetching %d days of BTC 1-minute candles from Coinbase...", args.days)
    candles = fetch_coinbase_candles(start_sec, end_sec)
    if candles.empty:
        logger.error("No candle data fetched — cannot proceed")
        sys.exit(1)

    logger.info("Generating synthetic Kalshi markets...")
    markets = generate_markets(candles)
    if not markets:
        logger.error("No markets generated — candle data may be insufficient")
        sys.exit(1)

    logger.info("Computing features for %d markets (%d snapshots each)...", len(markets), args.snapshots)
    rows = build_training_rows(candles, markets, snapshots_per_market=args.snapshots)
    if not rows:
        logger.error("No training rows produced")
        sys.exit(1)

    df = pd.DataFrame(rows)

    # Fill outcome_rate_bucket from historical rates
    df = compute_outcome_rate_buckets(df)

    # Reorder columns to match expected format
    output_cols = [c for c in OUTPUT_COLUMNS if c in df.columns]
    df = df[output_cols]

    # Sort chronologically
    df = df.sort_values(["close_ts", "market_ticker", "seconds_to_close"]).reset_index(drop=True)

    df.to_csv(args.output, index=False)
    logger.info("Saved %d rows × %d features to %s", len(df), len(df.columns), args.output)

    # Print summary stats
    logger.info("Class distribution:")
    counts = df["final_outcome_yes"].value_counts()
    for cls, count in counts.items():
        logger.info("  outcome=%s: %d (%.1f%%)", cls, count, count / len(df) * 100)

    logger.info("Entry bucket distribution:")
    print(df["entry_bucket"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
