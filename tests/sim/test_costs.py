# tests/sim/test_costs.py
import pytest
from sim.costs import CostModel, mark_price, entry_cost, exit_proceeds, fee


def test_invalid_side_raises():
    with pytest.raises(ValueError):
        mark_price("buy", 0.5)


def test_mark_price_sides():
    assert mark_price("yes", 0.40) == pytest.approx(0.40)
    assert mark_price("no", 0.40) == pytest.approx(0.60)


def test_entry_cost_adds_offsets():
    cfg = CostModel()
    # YES: mark 0.40 + 0.02
    assert entry_cost("yes", 0.40, cfg) == pytest.approx(0.42)
    # NO cheap: mark = 1-0.70 = 0.30 (<=0.40) -> +0.05
    assert entry_cost("no", 0.70, cfg) == pytest.approx(0.35)
    # NO expensive: mark = 1-0.30 = 0.70 (>0.40) -> +0.03
    assert entry_cost("no", 0.30, cfg) == pytest.approx(0.73)
    # capped at max_price
    assert entry_cost("yes", 0.99, cfg) == pytest.approx(0.99)


def test_exit_proceeds_subtracts_spread_and_floors_at_zero():
    cfg = CostModel()
    assert exit_proceeds("yes", 0.50, cfg) == pytest.approx(0.48)
    assert exit_proceeds("no", 0.90, cfg) == pytest.approx(0.08)  # mark .10 - .02
    assert exit_proceeds("yes", 0.01, cfg) == pytest.approx(0.0)  # floored
    assert exit_proceeds("no", 0.99, cfg) == pytest.approx(0.0)  # NO mark .01 - .02 floored


def test_fee_only_on_gains():
    cfg = CostModel()
    assert fee(10.0, cfg) == pytest.approx(0.10)
    assert fee(-10.0, cfg) == pytest.approx(0.0)
    assert fee(0.0, cfg) == pytest.approx(0.0)
