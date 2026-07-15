# tests/test_check_quality.py
import importlib.util
import pathlib

_spec = importlib.util.spec_from_file_location(
    "check_quality", pathlib.Path("scripts/check_quality.py")
)
cq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cq)


def test_regression_in_tests_fails():
    base = {"tests_passed": 100, "coverage_pct": 50.0}
    ok, new = cq.ratchet(99, 50.0, base)
    assert ok is False
    assert new == base  # unchanged on failure


def test_coverage_regression_fails():
    base = {"tests_passed": 100, "coverage_pct": 50.0}
    ok, new = cq.ratchet(100, 49.9, base)
    assert ok is False


def test_improvement_ratchets_up():
    base = {"tests_passed": 100, "coverage_pct": 50.0}
    ok, new = cq.ratchet(105, 55.0, base)
    assert ok is True
    assert new == {"tests_passed": 105, "coverage_pct": 55.0}


def test_equal_is_ok_and_holds_baseline():
    base = {"tests_passed": 100, "coverage_pct": 50.0}
    ok, new = cq.ratchet(100, 50.0, base)
    assert ok is True
    assert new == base


def test_check_only_never_raises_baseline():
    base = {"tests_passed": 100, "coverage_pct": 50.0}
    ok, new = cq.ratchet(105, 55.0, base, allow_raise=False)
    assert ok is True
    assert new == base  # not raised in check-only mode


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
