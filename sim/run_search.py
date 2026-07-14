"""CLI: run the strategy search end-to-end and write a report."""
from __future__ import annotations

import argparse

from sim.costs import CostModel
from sim.data import load_paths
from sim.validation import train_val_test_split
from sim.search import run_search, Gates
from sim.report import write_report
from sim.promote import config_to_settings, promotion_candidate

DEFAULT_SPACE = {
    "signal": ["ensemble", "mean_reversion"],
    "exit": ["hold", "tp_sl", "trailing"],
    "sizing": ["flat", "kelly", "payoff_aware"],
    "mispricing_threshold": [0.15, 0.20, 0.25],
    "max_entry_no": [0.55, 0.65, 0.80],
    "sl_abs": [0.0, 0.10, 0.15],
    "tp_abs": [0.0, 0.20],
    "trail_abs": [0.0, 0.10],
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    # Default is the per-poll log (multi-poll paths) so exit strategies can be backtested.
    parser.add_argument("--data", default="live_training_data_20260625.csv")
    parser.add_argument("--out", default="sim_results")
    args = parser.parse_args(argv)

    paths = load_paths(args.data, min_polls=1)
    train, val, test = train_val_test_split(paths)
    board = run_search(DEFAULT_SPACE, train, val, test, CostModel(), Gates())
    report = write_report(board, args.out)
    print(f"Report: {report['md']} | passed: {report['n_passed']}")

    cand = promotion_candidate(board)
    if cand is None:
        print("No promotion candidate — no strategy passed all gates.")
    else:
        print("Promotion candidate settings:", config_to_settings(cand["config"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
