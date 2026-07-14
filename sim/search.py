# sim/search.py
"""Grid search with a robust risk-adjusted objective and hard OOS gates."""
from __future__ import annotations

import itertools
from dataclasses import dataclass

from sim.costs import CostModel
from sim.engine import simulate
from sim.metrics import compute_metrics
from sim.signals import SIGNALS
from sim.exits import EXITS
from sim.sizing import SIZERS
from sim.validation import monte_carlo_pvalue, walk_forward


@dataclass
class Gates:
    min_trades: int = 30
    max_drawdown: float = 50.0
    mc_pvalue: float = 0.05
    min_folds_positive: int = 3


def build_grid(space: dict) -> list:
    keys = list(space.keys())
    return [dict(zip(keys, combo)) for combo in itertools.product(*space.values())]


def _defaults() -> dict:
    return dict(signal="ensemble", exit="hold", sizing="flat",
                mispricing_threshold=0.25, yes_cutoff=0.72, max_entry_yes=0.65,
                max_entry_no=0.80, no_max_p_raw=0.20, cutoff_buffer=0.03,
                min_entry_price=0.05, min_seconds=60, max_seconds=300,
                mr_return_5m=40.0, mr_price_floor=0.60,
                base_size=1.0, kelly_fraction=0.5, max_size=20.0,
                tp_abs=0.0, sl_abs=0.0, trail_abs=0.0)


def evaluate_config(cfg: dict, paths, cost_model: CostModel):
    full = _defaults()
    full.update(cfg)
    trades = simulate(paths, SIGNALS[full["signal"]], EXITS[full["exit"]],
                      SIZERS[full["sizing"]], full, cost_model)
    metrics = compute_metrics([t.pnl for t in trades],
                              [t.contracts for t in trades])
    return trades, metrics


def objective_score(metrics: dict) -> float:
    return metrics.get("sharpe", 0.0)


def passes_gates(trades, metrics, wf_positive_folds, wf_final_positive, mc, gates: Gates) -> bool:
    return (metrics["n_trades"] >= gates.min_trades
            and metrics["max_drawdown"] <= gates.max_drawdown
            and mc["p_value"] < gates.mc_pvalue
            and wf_positive_folds >= gates.min_folds_positive
            and wf_final_positive)


def run_search(space, train, val, test, cost_model, gates: Gates,
               n_folds: int = 4) -> list:
    board = []
    for cfg in build_grid(space):
        _, tr_metrics = evaluate_config(cfg, train, cost_model)
        val_trades, val_metrics = evaluate_config(cfg, val, cost_model)
        test_trades, test_metrics = evaluate_config(cfg, test, cost_model)

        folds = walk_forward(train, n_folds=n_folds)
        wf_pos = 0
        wf_final_positive = False
        for i, (f_train, f_test) in enumerate(folds):
            _, fm = evaluate_config(cfg, f_test, cost_model)
            if fm["total_pnl"] > 0:
                wf_pos += 1
            if i == len(folds) - 1:
                wf_final_positive = fm["total_pnl"] > 0

        # Gate on the VALIDATION partition; the TEST partition stays untouched by
        # selection and is reported only as the honest out-of-sample estimate.
        mc = monte_carlo_pvalue(val_trades)
        passed = passes_gates(val_trades, val_metrics, wf_pos, wf_final_positive, mc, gates)
        board.append({
            "config": cfg,
            "train_metrics": tr_metrics,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "mc_pvalue": mc["p_value"],
            "wf_positive_folds": wf_pos,
            "wf_final_positive": wf_final_positive,
            "passed": passed,
            "score": objective_score(val_metrics),
        })
    board.sort(key=lambda r: (r["passed"], r["score"],
                              r["val_metrics"]["profit_factor"]), reverse=True)
    return board
