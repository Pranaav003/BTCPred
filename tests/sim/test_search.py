# tests/sim/test_search.py
import pytest
from sim.data import MarketPath, Poll
from sim.costs import CostModel
from sim.search import (build_grid, evaluate_config, objective_score,
                        passes_gates, run_search, Gates)


def test_build_grid_cartesian():
    grid = build_grid({"a": [1, 2], "b": [3]})
    assert grid == [{"a": 1, "b": 3}, {"a": 2, "b": 3}]


def _winning_paths(n):
    # bearish NO signals that all resolve NO -> reliable winners
    return [MarketPath(f"M{i}", 300, 1000 + i * 10_000, 0,
                       [Poll(200, 0.40, 0.10, {})]) for i in range(n)]


def _base_config():
    return dict(signal="ensemble", exit="hold", sizing="flat",
                mispricing_threshold=0.25, yes_cutoff=0.72, max_entry_yes=0.65,
                max_entry_no=0.80, no_max_p_raw=0.20, cutoff_buffer=0.03,
                min_entry_price=0.05, min_seconds=60, max_seconds=300,
                mr_return_5m=40.0, mr_price_floor=0.60,
                base_size=1.0, kelly_fraction=0.5, max_size=20.0,
                tp_abs=0.0, sl_abs=0.0, trail_abs=0.0)


def test_evaluate_config_runs():
    trades, metrics = evaluate_config(_base_config(), _winning_paths(40),
                                      CostModel())
    assert metrics["n_trades"] == 40 and metrics["win_rate"] == 1.0


def test_objective_score_is_sharpe():
    assert objective_score({"sharpe": 1.23}) == 1.23


def test_passes_gates_rejects_thin_sample():
    gates = Gates(min_trades=30)
    assert passes_gates([], {"n_trades": 5, "max_drawdown": 0.0}, 4,
                        {"p_value": 0.01}, gates) is False


def test_run_search_ranks_and_gates():
    paths = _winning_paths(60)
    space = {"signal": ["ensemble"], "exit": ["hold"], "sizing": ["flat"],
             "mispricing_threshold": [0.25], "max_entry_no": [0.80]}
    board = run_search(space, paths, paths, paths, CostModel(), Gates(),
                       n_folds=4)
    assert len(board) == 1
    row = board[0]
    assert "config" in row and "score" in row and "passed" in row
    assert row["test_metrics"]["n_trades"] > 0
