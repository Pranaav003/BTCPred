# tests/test_check_quality.py
import importlib.util
import pathlib

_spec = importlib.util.spec_from_file_location(
    "check_quality", pathlib.Path("scripts/check_quality.py")
)
cq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cq)


_DIRS = {"tests_passed": "up", "coverage_pct": "up",
         "ruff_violations": "down", "mypy_errors": "down"}


def _base():
    return {"tests_passed": 204, "coverage_pct": 40.0, "ruff_violations": 50, "mypy_errors": 20}


def test_ratchet_all_hold_when_equal():
    ok, new = cq.ratchet(_base(), _base())
    assert ok is True and new == _base()


def test_ratchet_all_improves_every_metric():
    cur = {"tests_passed": 210, "coverage_pct": 45.0, "ruff_violations": 30, "mypy_errors": 10}
    ok, new = cq.ratchet(cur, _base())
    assert ok is True and new == cur  # all moved in the improving direction


def test_ratchet_fails_on_test_regression():
    cur = {"tests_passed": 203, "coverage_pct": 40.0, "ruff_violations": 50, "mypy_errors": 20}
    ok, new = cq.ratchet(cur, _base())
    assert ok is False and new == _base()


def test_ratchet_fails_on_more_ruff_violations():
    cur = {"tests_passed": 204, "coverage_pct": 40.0, "ruff_violations": 51, "mypy_errors": 20}
    ok, new = cq.ratchet(cur, _base())
    assert ok is False


def test_ratchet_check_only_never_moves():
    cur = {"tests_passed": 210, "coverage_pct": 45.0, "ruff_violations": 30, "mypy_errors": 10}
    ok, new = cq.ratchet(cur, _base(), allow_move=False)
    assert ok is True and new == _base()


def test_ratchet_directional_up():
    assert cq.ratchet_directional(205, 204, "up") == (True, 205)   # improve
    assert cq.ratchet_directional(204, 204, "up") == (True, 204)   # equal
    assert cq.ratchet_directional(203, 204, "up") == (False, 204)  # regress -> hold


def test_ratchet_directional_down():
    assert cq.ratchet_directional(5, 8, "down") == (True, 5)    # fewer violations = improve
    assert cq.ratchet_directional(8, 8, "down") == (True, 8)    # equal
    assert cq.ratchet_directional(9, 8, "down") == (False, 8)   # more = regress -> hold


import subprocess as _sp


def test_run_ruff_counts_found_errors(monkeypatch):
    def fake_run(*a, **k):
        return _sp.CompletedProcess(a, 1, stdout="app/x.py:1:1: F401 unused\nFound 3 errors.\n", stderr="")
    monkeypatch.setattr(cq.subprocess, "run", fake_run)
    assert cq._run_ruff() == 3


def test_run_ruff_zero_on_clean(monkeypatch):
    monkeypatch.setattr(cq.subprocess, "run",
                        lambda *a, **k: _sp.CompletedProcess(a, 0, stdout="All checks passed!\n", stderr=""))
    assert cq._run_ruff() == 0


def test_run_mypy_counts_found_errors(monkeypatch):
    def fake_run(*a, **k):
        return _sp.CompletedProcess(a, 1, stdout="app/x.py:1: error: bad\nFound 2 errors in 1 file\n", stderr="")
    monkeypatch.setattr(cq.subprocess, "run", fake_run)
    assert cq._run_mypy() == 2


def test_run_ruff_sentinel_on_unparseable(monkeypatch):
    # returncode != 0 but no "Found N" and no violation lines -> sentinel (regression), never 0
    monkeypatch.setattr(cq.subprocess, "run",
                        lambda *a, **k: _sp.CompletedProcess(a, 2, stdout="ruff: internal error\n", stderr=""))
    assert cq._run_ruff() == cq._SENTINEL


def test_run_mypy_sentinel_on_unparseable(monkeypatch):
    # returncode != 0 with no parseable count -> sentinel (regression), never 0
    monkeypatch.setattr(cq.subprocess, "run",
                        lambda *a, **k: _sp.CompletedProcess(a, 2, stdout="mypy: internal crash\n", stderr=""))
    assert cq._run_mypy() == cq._SENTINEL


def test_perf_ok_band():
    assert cq.perf_ok(44.0, 43.0) is True     # within +10%
    assert cq.perf_ok(47.3, 43.0) is True      # ~ exactly 43*1.1
    assert cq.perf_ok(48.0, 43.0) is False     # over band


def test_perf_ratchet_within_band_ratchets_down():
    base = {"predict_proba_raw_ms": 43.0, "api_health_ms": 0.2}
    ok, new = cq.perf_ratchet({"predict_proba_raw_ms": 40.0, "api_health_ms": 0.2}, base)
    assert ok is True
    assert new["predict_proba_raw_ms"] == 40.0   # improved -> min
    assert new["api_health_ms"] == 0.2


def test_perf_ratchet_over_band_fails_and_holds():
    base = {"predict_proba_raw_ms": 43.0}
    ok, new = cq.perf_ratchet({"predict_proba_raw_ms": 48.0}, base)  # >43*1.1
    assert ok is False and new == base


def test_perf_ratchet_seeds_unknown_metric():
    ok, new = cq.perf_ratchet({"new_ms": 5.0}, {})
    assert ok is True and new["new_ms"] == 5.0


def test_perf_cli_within_band_returns_zero(monkeypatch, tmp_path):
    import json
    bl = tmp_path / "tb.json"
    bl.write_text(json.dumps({"tests_passed": 1, "coverage_pct": 0.0,
                              "ruff_violations": 0, "mypy_errors": 0,
                              "perf": {"api_health_ms": 0.2}}))
    monkeypatch.setattr(cq, "BASELINE_PATH", bl)
    monkeypatch.setattr(cq, "_run_perf", lambda: {"api_health_ms": 0.19})
    assert cq.main(["--perf"]) == 0
    assert json.loads(bl.read_text())["perf"]["api_health_ms"] == 0.19  # ratcheted down


def test_perf_cli_over_band_returns_one(monkeypatch, tmp_path):
    import json
    bl = tmp_path / "tb.json"
    bl.write_text(json.dumps({"tests_passed": 1, "coverage_pct": 0.0,
                              "ruff_violations": 0, "mypy_errors": 0,
                              "perf": {"api_health_ms": 0.2}}))
    monkeypatch.setattr(cq, "BASELINE_PATH", bl)
    monkeypatch.setattr(cq, "_run_perf", lambda: {"api_health_ms": 0.8})  # over relative+floor band (0.2+0.5=0.7)
    assert cq.main(["--perf"]) == 1
    assert json.loads(bl.read_text())["perf"]["api_health_ms"] == 0.2  # unchanged


def test_perf_ok_absolute_floor_tolerates_subms_noise():
    # 0.125ms baseline: +10% alone = 0.1375 (too tight); the 0.5ms floor gives real headroom.
    assert cq.perf_ok(0.14, 0.125) is True
    assert cq.perf_ok(0.60, 0.125) is True
    assert cq.perf_ok(0.70, 0.125) is False   # beyond the floor


def test_perf_ok_large_metric_uses_relative_band():
    assert cq.perf_ok(45.0, 43.0) is True      # within +10%
    assert cq.perf_ok(48.0, 43.0) is False     # over +10% (floor 43.5 doesn't rescue)


def test_init_preserves_perf_block(monkeypatch, tmp_path):
    import json
    bl = tmp_path / "tb.json"
    bl.write_text(json.dumps({"tests_passed": 1, "coverage_pct": 0.0,
                              "ruff_violations": 0, "mypy_errors": 0,
                              "perf": {"api_health_ms": 0.2}}))
    monkeypatch.setattr(cq, "BASELINE_PATH", bl)
    monkeypatch.setattr(cq, "_measure",
                        lambda: (True, {"tests_passed": 5, "coverage_pct": 10.0,
                                        "ruff_violations": 3, "mypy_errors": 1}))
    assert cq.main(["--init"]) == 0
    saved = json.loads(bl.read_text())
    assert saved["perf"] == {"api_health_ms": 0.2}   # preserved, not clobbered
    assert saved["tests_passed"] == 5


def test_init_preserves_coverage_and_mypy_floors(monkeypatch, tmp_path):
    # Refutation-derived: --init must NOT silently lower the coverage/mypy floors
    # (only ruff_violations is expected to rise when a new rule is enabled). It
    # preserves the stricter of prior/current for coverage (max) and mypy (min),
    # exactly as it already preserves the perf block.
    import json
    bl = tmp_path / "tb.json"
    bl.write_text(json.dumps({"tests_passed": 300, "coverage_pct": 44.0,
                              "ruff_violations": 274, "mypy_errors": 38,
                              "perf": {"api_health_ms": 0.2}}))
    monkeypatch.setattr(cq, "BASELINE_PATH", bl)
    # New run: coverage dipped (new unexercised code), mypy improved, ruff rose.
    monkeypatch.setattr(cq, "_measure",
                        lambda: (True, {"tests_passed": 360, "coverage_pct": 42.5,
                                        "ruff_violations": 307, "mypy_errors": 35}))
    assert cq.main(["--init"]) == 0
    saved = json.loads(bl.read_text())
    assert saved["coverage_pct"] == 44.0   # floor preserved (not lowered to 42.5)
    assert saved["mypy_errors"] == 35      # stricter (fewer) kept
    assert saved["ruff_violations"] == 307  # ruff re-seeded upward (rule addition)
    assert saved["tests_passed"] == 360
    assert saved["perf"] == {"api_health_ms": 0.2}  # still preserved


def test_init_seeds_fresh_when_no_prior_baseline(monkeypatch, tmp_path):
    # With no prior baseline, --init just stamps current values (nothing to preserve).
    import json
    bl = tmp_path / "tb.json"  # does not exist
    monkeypatch.setattr(cq, "BASELINE_PATH", bl)
    monkeypatch.setattr(cq, "_measure",
                        lambda: (True, {"tests_passed": 10, "coverage_pct": 50.0,
                                        "ruff_violations": 5, "mypy_errors": 2}))
    assert cq.main(["--init"]) == 0
    saved = json.loads(bl.read_text())
    assert saved["coverage_pct"] == 50.0 and saved["ruff_violations"] == 5


def test_perf_ratchet_mixed_pass_and_regress_holds_all():
    # one metric improves, another regresses -> whole run fails, baseline unchanged (no partial ratchet)
    base = {"a_ms": 10.0, "b_ms": 10.0}
    ok, new = cq.perf_ratchet({"a_ms": 8.0, "b_ms": 20.0}, base)
    assert ok is False and new == base
