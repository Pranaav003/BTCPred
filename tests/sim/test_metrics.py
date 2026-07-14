# tests/sim/test_metrics.py
import math
import pytest
from sim.metrics import breakeven_win_rate, compute_metrics, calibration


def test_breakeven_win_rate():
    # avg win 0.30, avg loss 0.62 -> 0.62 / 0.92
    assert breakeven_win_rate(0.30, 0.62) == pytest.approx(0.62 / 0.92)


def test_compute_metrics_basic():
    pnls = [1.0, 1.0, -2.0, 1.0]  # 3 wins, 1 loss
    contracts = [1, 1, 1, 1]
    m = compute_metrics(pnls, contracts)
    assert m["n_trades"] == 4
    assert m["win_rate"] == pytest.approx(0.75)
    assert m["total_pnl"] == pytest.approx(1.0)
    assert m["avg_win"] == pytest.approx(1.0)
    assert m["avg_loss"] == pytest.approx(-2.0)
    assert m["profit_factor"] == pytest.approx(3.0 / 2.0)
    assert m["ev_per_contract"] == pytest.approx(1.0 / 4)
    # max drawdown: equity 1,2,0,1 -> peak 2 then 0 => dd 2.0
    assert m["max_drawdown"] == pytest.approx(2.0)


def test_compute_metrics_empty():
    m = compute_metrics([], [])
    assert m["n_trades"] == 0
    assert m["total_pnl"] == 0.0
    assert m["profit_factor"] == 0.0


def test_calibration_brier():
    # perfect predictions -> brier 0
    assert calibration([1.0, 0.0], [1, 0])["brier"] == pytest.approx(0.0)
    # coin-flip on a certain event
    assert calibration([0.5, 0.5], [1, 0])["brier"] == pytest.approx(0.25)
