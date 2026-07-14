# tests/sim/test_report.py
import os
from sim.report import leaderboard_to_dataframe, write_report


def _row(passed, score, pnl):
    return {
        "config": {"signal": "ensemble", "exit": "hold", "sizing": "flat"},
        "train_metrics": {}, "val_metrics": {},
        "test_metrics": {"n_trades": 40, "win_rate": 0.8, "total_pnl": pnl,
                         "profit_factor": 2.0, "max_drawdown": 3.0},
        "mc_pvalue": 0.01, "wf_positive_folds": 4,
        "passed": passed, "score": score,
    }


def test_leaderboard_dataframe_columns():
    df = leaderboard_to_dataframe([_row(True, 1.0, 50.0)])
    assert df.iloc[0]["signal"] == "ensemble"
    assert df.iloc[0]["test_total_pnl"] == 50.0
    assert df.iloc[0]["passed"] == True  # noqa: E712


def test_write_report_with_passing(tmp_path):
    out = write_report([_row(True, 1.0, 50.0)], str(tmp_path))
    assert os.path.exists(out["csv"]) and os.path.exists(out["md"])
    assert out["n_passed"] == 1
    assert "ensemble" in open(out["md"]).read()


def test_write_report_no_passing_states_it(tmp_path):
    out = write_report([_row(False, -1.0, -50.0)], str(tmp_path))
    assert out["n_passed"] == 0
    assert "No strategy passed all gates" in open(out["md"]).read()
