# tests/test_precommit_config.py
import pathlib


def test_precommit_config_runs_quality_gate():
    p = pathlib.Path(".pre-commit-config.yaml")
    assert p.exists()
    text = p.read_text()
    assert "repos:" in text                          # it is a pre-commit config
    assert "check_quality.py --check-only" in text   # and it runs our gate


def test_precommit_has_manual_perf_hook():
    text = pathlib.Path(".pre-commit-config.yaml").read_text()
    assert "check_quality.py --perf" in text
    assert "manual" in text  # perf hook is manual-stage, not the default commit gate
