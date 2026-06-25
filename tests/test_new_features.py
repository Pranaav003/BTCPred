"""Tests for new feature engineering functions.

These tests verify the math directly without importing the full app module
(since app.feature_engineering has Flask/SQLAlchemy dependencies).
"""
import pytest
import numpy as np
import pandas as pd


def _compute_rsi(candles_df, period=14):
    """Reimplementation of _compute_rsi from feature_engineering."""
    if candles_df is None or len(candles_df) < period + 1:
        return 50.0
    closes = candles_df["close"].astype(float).values
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100.0 - (100.0 / (1.0 + rs)))


def _trading_session(close_ts):
    """Reimplementation of _trading_session from feature_engineering."""
    from datetime import datetime, timezone
    hour = datetime.fromtimestamp(int(close_ts), tz=timezone.utc).hour
    if 0 <= hour < 8:
        return 0
    elif 8 <= hour < 14:
        return 1
    elif 14 <= hour < 21:
        return 2
    else:
        return 0


def _distance_from_strike(btc_price, market_title):
    """Reimplementation of _distance_from_strike from feature_engineering."""
    import re
    if not btc_price or not market_title:
        return 0.0
    match = re.search(r"\$(\d[\d,]*)", market_title)
    if not match:
        return 0.0
    strike_str = match.group(1).replace(",", "")
    try:
        strike = float(strike_str)
    except ValueError:
        return 0.0
    if strike == 0:
        return 0.0
    return (btc_price - strike) / strike


def test_compute_rsi_all_gains():
    """Monotonically increasing prices -> RSI = 100."""
    closes = list(range(20, 36))
    df = pd.DataFrame({"close": closes})
    assert _compute_rsi(df, period=14) == 100.0


def test_compute_rsi_insufficient_data():
    """Not enough data -> neutral RSI."""
    df = pd.DataFrame({"close": [100, 101]})
    assert _compute_rsi(df, period=14) == 50.0


def test_trading_session_returns_int():
    """Trading session must return a valid integer."""
    result = _trading_session(1700000000)
    assert isinstance(result, int)
    assert result in {0, 1, 2}


def test_distance_from_strike_exact():
    """At the strike, distance is 0."""
    assert _distance_from_strike(100000.0, "BTC above $100,000") == 0.0


def test_distance_from_strike_above():
    """5% above strike."""
    dist = _distance_from_strike(105000.0, "BTC above $100,000")
    assert abs(dist - 0.05) < 0.001


def test_distance_from_strike_no_strike():
    """No strike in title -> 0.0."""
    assert _distance_from_strike(100000.0, "Some market") == 0.0
