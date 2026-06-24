"""Tests for RAW_FEATURES deduplication and correctness."""
from train_raw_model import RAW_FEATURES


def test_no_duplicate_features():
    """RAW_FEATURES must not contain duplicate feature names."""
    assert len(RAW_FEATURES) == len(set(RAW_FEATURES)), (
        f"Duplicate features found: {[f for f in RAW_FEATURES if RAW_FEATURES.count(f) > 1]}"
    )


def test_no_known_redundant_features():
    """Features that are mathematically identical to others must be removed."""
    redundant = {"momentum_1m", "momentum_3m", "momentum_5m", "price_velocity_5m"}
    present = redundant & set(RAW_FEATURES)
    assert len(present) == 0, f"Redundant features still present: {present}"


def test_momentum_acceleration_is_present():
    """momentum_acceleration is genuinely different and must be kept."""
    assert "momentum_acceleration" in RAW_FEATURES
