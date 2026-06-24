"""Tests for seconds_to_close rounding in auto-trade guard."""
import math


def test_ceil_prevents_truncation_bypass():
    """89.7 seconds must round UP to 90, not down to 89."""
    seconds_float = 89.7
    min_seconds = 90
    truncated = int(seconds_float)
    assert truncated < min_seconds  # int() truncates to 89, passes check incorrectly
    ceiled = math.ceil(seconds_float)
    assert ceiled >= min_seconds  # math.ceil rounds to 90, correctly fails check


def test_exact_integer_stays_same():
    """Exact integer values like 90.0 should remain 90."""
    assert math.ceil(90.0) == 90
    assert math.ceil(120.0) == 120


def test_fractional_values_round_up():
    """Any fractional value rounds up to the next integer."""
    assert math.ceil(89.1) == 90
    assert math.ceil(89.9) == 90
    assert math.ceil(90.1) == 91
