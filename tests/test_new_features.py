"""Tests for new feature engineering functions."""
import pytest
import numpy as np
import pandas as pd


def test_compute_rsi_all_gains():
    """Monotonically increasing prices -> RSI = 100."""
    from app.feature_engineering import _compute_rsi
    closes = list(range(20, 36))
    df = pd.DataFrame({"close": closes})
    assert _compute_rsi(df, period=14) == 100.0


def test_compute_rsi_insufficient_data():
    """Not enough data -> neutral RSI."""
    from app.feature_engineering import _compute_rsi
    df = pd.DataFrame({"close": [100, 101]})
    assert _compute_rsi(df, period=14) == 50.0


def test_trading_session_returns_int():
    """Trading session must return an integer."""
    from app.feature_engineering import _trading_session
    result = _trading_session(1700000000)
    assert isinstance(result, int)
    assert result in {0, 1, 2}


def test_distance_from_strike_exact():
    """At the strike, distance is 0."""
    from app.feature_engineering import _distance_from_strike
    assert _distance_from_strike(100000.0, "BTC above $100,000") == 0.0


def test_distance_from_strike_above():
    """5% above strike."""
    from app.feature_engineering import _distance_from_strike
    dist = _distance_from_strike(105000.0, "BTC above $100,000")
    assert abs(dist - 0.05) < 0.001


def test_distance_from_strike_no_strike():
    """No strike in title -> 0.0."""
    from app.feature_engineering import _distance_from_strike
    assert _distance_from_strike(100000.0, "Some market") == 0.0
