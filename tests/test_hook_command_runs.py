# tests/test_hook_command_runs.py
import importlib.util
import json
import pathlib


def _load_cq():
    spec = importlib.util.spec_from_file_location(
        "check_quality", pathlib.Path("scripts/check_quality.py"))
    cq = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cq)
    return cq


def test_settings_json_has_stop_hook():
    p = pathlib.Path(".claude/settings.json")
    assert p.exists()
    cfg = json.loads(p.read_text())
    hooks = cfg.get("hooks", {})
    assert "Stop" in hooks
    blob = json.dumps(hooks["Stop"])
    assert "check_quality.py" in blob


def test_check_only_returns_zero_when_suite_green(monkeypatch):
    cq = _load_cq()
    monkeypatch.setattr(cq, "_measure", lambda: (True, {
        "tests_passed": 10_000, "coverage_pct": 100.0,
        "ruff_violations": 0, "mypy_errors": 0}))
    assert cq.main(["--check-only"]) == 0


def test_check_only_returns_one_when_suite_red(monkeypatch):
    cq = _load_cq()
    # all_passed=False makes main return 1 early, so the metric values are irrelevant.
    monkeypatch.setattr(cq, "_measure", lambda: (False, {
        "tests_passed": 0, "coverage_pct": 0.0,
        "ruff_violations": 0, "mypy_errors": 0}))
    assert cq.main(["--check-only"]) == 1
