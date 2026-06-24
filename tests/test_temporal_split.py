"""Tests for market-level temporal split without data leakage."""
import pandas as pd
import numpy as np


def _make_dataset(n_markets=10, rows_per_market=5):
    """Create a synthetic dataset with known market_ticker values."""
    rows = []
    for i in range(n_markets):
        ticker = f"KXBTC15M-{i:03d}"
        base_ts = 1700000000 + i * 900  # 15-min apart
        for j in range(rows_per_market):
            rows.append({
                "market_ticker": ticker,
                "close_ts": base_ts + j * 60,
                "final_outcome_yes": np.random.choice([0, 1]),
                "return_1m": np.random.randn() * 0.01,
                "return_3m": np.random.randn() * 0.02,
                "return_5m": np.random.randn() * 0.03,
            })
    return pd.DataFrame(rows)


def test_no_market_in_both_train_and_test():
    """No market_ticker may appear in both train and test sets."""
    from train_raw_model import temporal_split
    df = _make_dataset(n_markets=20, rows_per_market=3)
    train_df, test_df = temporal_split(df)
    train_tickers = set(train_df["market_ticker"].unique())
    test_tickers = set(test_df["market_ticker"].unique())
    overlap = train_tickers & test_tickers
    assert len(overlap) == 0, f"Markets in both train and test: {overlap}"


def test_embargo_period_between_train_and_test():
    """Test set must start at least 900s (15 min) after the last training close_ts."""
    from train_raw_model import temporal_split
    df = _make_dataset(n_markets=20, rows_per_market=3)
    train_df, test_df = temporal_split(df)
    max_train_ts = train_df["close_ts"].max()
    min_test_ts = test_df["close_ts"].min()
    gap = min_test_ts - max_train_ts
    assert gap >= 900, f"Embargo gap is only {gap}s, need >= 900s (15 min)"


def test_train_comes_before_test():
    """All training data must be chronologically before test data."""
    from train_raw_model import temporal_split
    df = _make_dataset(n_markets=20, rows_per_market=3)
    train_df, test_df = temporal_split(df)
    max_train_ts = train_df["close_ts"].max()
    min_test_ts = test_df["close_ts"].min()
    assert max_train_ts < min_test_ts
