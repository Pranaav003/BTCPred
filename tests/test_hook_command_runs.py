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
    # Validate the hook command's LOGIC without actually running pytest again.
    # (Running the real suite here would recurse: the subprocess pytest would
    # re-collect this test, spawning pytest endlessly.)
    cq = _load_cq()
    monkeypatch.setattr(cq, "_run_suite", lambda: (True, 10_000, 100.0))
    assert cq.main(["--check-only"]) == 0


def test_check_only_returns_one_when_suite_red(monkeypatch):
    cq = _load_cq()
    monkeypatch.setattr(cq, "_run_suite", lambda: (False, 0, 0.0))
    assert cq.main(["--check-only"]) == 1
