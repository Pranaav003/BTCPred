# tests/sim/test_sizing.py
import pytest
from sim.sizing import flat_size, fractional_kelly_size, payoff_aware_size, SIZERS


def _cfg(**o):
    b = dict(base_size=5.0, kelly_fraction=0.5, max_size=20.0)
    b.update(o)
    return b


def test_flat_size_is_constant():
    assert flat_size(0.1, 0.6, 0.42, _cfg()) == 5.0


def test_kelly_scales_with_edge_and_caps():
    # win_prob 0.7, entry_cost 0.5 -> payoff b = (1-0.5)/0.5 = 1.0
    # kelly f = (b*p - (1-p))/b = (0.7 - 0.3)/1 = 0.4 ; half-kelly -> 0.2 of base*...
    size = fractional_kelly_size(0.2, 0.7, 0.5, _cfg())
    assert size > 0
    # never exceeds max_size
    assert fractional_kelly_size(0.9, 0.99, 0.01, _cfg()) <= 20.0


def test_kelly_zero_when_no_edge():
    # win_prob below breakeven -> non-positive kelly -> 0
    assert fractional_kelly_size(0.0, 0.30, 0.5, _cfg()) == 0.0


def test_payoff_aware_skips_below_breakeven():
    # entry_cost 0.78 -> if lose, lose 0.78; if win, gain 0.22. breakeven wr = .78
    # win_prob 0.60 < breakeven -> skip (0.0)
    assert payoff_aware_size(0.1, 0.60, 0.78, _cfg()) == 0.0
    # win_prob 0.85 > breakeven -> positive size
    assert payoff_aware_size(0.1, 0.85, 0.78, _cfg()) > 0.0


def test_registry():
    assert set(SIZERS) == {"flat", "kelly", "payoff_aware"}
