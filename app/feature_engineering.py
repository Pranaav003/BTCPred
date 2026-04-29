"""Feature engineering from raw candle/trade data for model inference."""

from __future__ import annotations

import concurrent.futures
import time
from typing import Any

import pandas as pd

from app.kalshi_client import PREFERRED_STRIKE, get_active_market, get_candles, get_trades


def _safe_std(series: pd.Series) -> float:
    """Return population std for numeric values, else 0.0."""
    if series is None:
        return 0.0
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if len(numeric) < 2:
        return 0.0
    return float(numeric.std(ddof=0))


def _weighted_avg_price(trades_df: pd.DataFrame) -> float | None:
    """Return qty-weighted average trade price with safe fallbacks."""
    if trades_df is None or trades_df.empty:
        return None

    prices = pd.to_numeric(trades_df.get("price"), errors="coerce")
    qty = pd.to_numeric(trades_df.get("qty"), errors="coerce")
    valid = pd.DataFrame({"price": prices, "qty": qty}).dropna(subset=["price"])
    if valid.empty:
        return None

    valid["qty"] = valid["qty"].fillna(0.0)
    total_qty = float(valid["qty"].sum())
    if total_qty == 0.0:
        return float(valid["price"].mean())

    return float((valid["price"] * valid["qty"]).sum() / total_qty)


def _flip_count(closes: pd.Series | list[float]) -> int:
    """Count sign reversals of consecutive close differences."""
    series = pd.Series(closes, dtype="float64").dropna()
    if len(series) < 3:
        return 0

    diffs = series.diff().dropna()
    if diffs.empty:
        return 0
    signs = diffs.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

    flips = 0
    prev_sign = 0
    for sign in signs:
        if sign == 0:
            continue
        if prev_sign != 0 and sign != prev_sign:
            flips += 1
        prev_sign = sign
    return int(flips)


def _closest_bucket(seconds: int) -> int:
    """Return nearest bucket among [60, 120, 180, 300]."""
    buckets = [60, 120, 180, 300]
    return min(buckets, key=lambda b: abs(b - int(seconds)))


def _price_at_offset(candles_df: pd.DataFrame, snapshot_ts: int, offset_seconds: int) -> float | None:
    """Get most recent close at/before snapshot_ts - offset_seconds."""
    if candles_df is None or candles_df.empty:
        return None
    target_ts = int(snapshot_ts) - int(offset_seconds)
    eligible = candles_df[pd.to_numeric(candles_df.get("ts"), errors="coerce") <= target_ts]
    if eligible.empty:
        return None
    row = eligible.sort_values("ts").iloc[-1]
    close = pd.to_numeric(pd.Series([row.get("close")]), errors="coerce").iloc[0]
    if pd.isna(close):
        return None
    return float(close)


def _window(df: pd.DataFrame, start_ts: int, end_ts: int) -> pd.DataFrame:
    """Filter rows where ts > start_ts and ts <= end_ts."""
    if df is None or df.empty:
        return pd.DataFrame(columns=list(df.columns) if df is not None else [])
    ts = pd.to_numeric(df.get("ts"), errors="coerce")
    return df[(ts > int(start_ts)) & (ts <= int(end_ts))].copy()


def compute_features(market_dict: dict[str, Any]) -> dict | None:
    """Compute all model features from live market candles/trades."""
    ticker = market_dict.get("ticker")
    close_ts = market_dict.get("close_ts")
    title = market_dict.get("title")
    close_time_iso = market_dict.get("close_time_iso")

    if ticker is None or close_ts is None:
        return None

    snapshot_ts = int(time.time())
    seconds_to_close = int(close_ts) - snapshot_ts
    if seconds_to_close <= 0:
        return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        candle_future = ex.submit(get_candles, str(ticker), int(close_ts))
        trade_future = ex.submit(get_trades, str(ticker), snapshot_ts - 600, snapshot_ts)
        candles = candle_future.result(timeout=15)
        trades = trade_future.result(timeout=15)

    if candles is None or candles.empty:
        return None

    if trades is None:
        trades = pd.DataFrame(columns=["ts", "price", "qty"])

    price_now = _price_at_offset(candles, snapshot_ts, 0)
    if price_now is None:
        return None

    price_1m = _price_at_offset(candles, snapshot_ts, 60)
    price_3m = _price_at_offset(candles, snapshot_ts, 180)
    price_5m = _price_at_offset(candles, snapshot_ts, 300)

    price_1m = price_now if price_1m is None else price_1m
    price_3m = price_now if price_3m is None else price_3m
    price_5m = price_now if price_5m is None else price_5m

    return_1m = float(price_now - price_1m)
    return_3m = float(price_now - price_3m)
    return_5m = float(price_now - price_5m)

    c3 = _window(candles, snapshot_ts - 180, snapshot_ts)
    c5 = _window(candles, snapshot_ts - 300, snapshot_ts)

    t1 = _window(trades, snapshot_ts - 60, snapshot_ts)
    t3 = _window(trades, snapshot_ts - 180, snapshot_ts)
    t5 = _window(trades, snapshot_ts - 300, snapshot_ts)

    volatility_3m = _safe_std(c3.get("close", pd.Series(dtype="float64")))
    volatility_5m = _safe_std(c5.get("close", pd.Series(dtype="float64")))

    if c5.empty:
        range_5m = 0.0
        flip_count_5m = 0
    else:
        high_vals = pd.to_numeric(c5.get("high"), errors="coerce")
        low_vals = pd.to_numeric(c5.get("low"), errors="coerce")
        high_max = high_vals.max() if not high_vals.dropna().empty else None
        low_min = low_vals.min() if not low_vals.dropna().empty else None
        range_5m = float((high_max - low_min) if high_max is not None and low_min is not None else 0.0)
        flip_count_5m = _flip_count(pd.to_numeric(c5.get("close"), errors="coerce"))

    trade_count_1m = int(len(t1))
    trade_count_3m = int(len(t3))
    trade_count_5m = int(len(t5))

    volume_1m = float(pd.to_numeric(t1.get("qty"), errors="coerce").fillna(0.0).sum()) if not t1.empty else 0.0
    volume_3m = float(pd.to_numeric(t3.get("qty"), errors="coerce").fillna(0.0).sum()) if not t3.empty else 0.0
    volume_5m = float(pd.to_numeric(t5.get("qty"), errors="coerce").fillna(0.0).sum()) if not t5.empty else 0.0

    avg_trade_price_1m = _weighted_avg_price(t1)
    avg_trade_price_3m = _weighted_avg_price(t3)
    avg_trade_price_1m = price_now if avg_trade_price_1m is None else float(avg_trade_price_1m)
    avg_trade_price_3m = price_now if avg_trade_price_3m is None else float(avg_trade_price_3m)

    momentum_1m = return_1m
    momentum_3m = return_3m
    momentum_5m = return_5m
    momentum_acceleration = float(return_1m - (return_3m / 3.0))
    price_velocity_5m = float(return_5m / 5.0)

    inv_time = 1.0 / max(seconds_to_close, 1)
    return_1m_x_inv_time = float(return_1m * inv_time)
    return_3m_x_inv_time = float(return_3m * inv_time)
    volatility_5m_x_inv_time = float(volatility_5m * inv_time)
    vol_score = min(float(volatility_5m) / 0.15, 1.0)
    flip_score = min(float(flip_count_5m) / 5.0, 1.0)
    same_direction = (
        (float(return_1m) > 0 and float(return_5m) > 0)
        or (float(return_1m) < 0 and float(return_5m) < 0)
    )
    direction_score = 0.2 if same_direction else 0.8
    reversal_risk = (vol_score * 0.4) + (flip_score * 0.3) + (direction_score * 0.3)

    entry_bucket = _closest_bucket(seconds_to_close)

    return {
        "market_ticker": str(ticker),
        "market_title": title,
        "close_time_iso": close_time_iso,
        "close_ts": int(close_ts),
        "snapshot_ts": snapshot_ts,
        "seconds_to_close": int(seconds_to_close),
        "entry_bucket": int(entry_bucket),
        "price_now": float(price_now),
        "p_market": float(price_now),
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
        "momentum_1m": float(momentum_1m),
        "momentum_3m": float(momentum_3m),
        "momentum_5m": float(momentum_5m),
        "momentum_acceleration": float(momentum_acceleration),
        "price_velocity_5m": float(price_velocity_5m),
        "flip_count_5m": float(flip_count_5m),
        "reversal_risk": float(reversal_risk),
        "return_1m_x_inv_time": float(return_1m_x_inv_time),
        "return_3m_x_inv_time": float(return_3m_x_inv_time),
        "volatility_5m_x_inv_time": float(volatility_5m_x_inv_time),
    }


def get_live_snapshot() -> dict | None:
    """Fetch active market and return computed live feature snapshot."""
    market = get_active_market(preferred_suffix=PREFERRED_STRIKE)
    if market is None:
        return None
    return compute_features(market)
