# tests/test_precommit_config.py
import pathlib


def test_precommit_config_runs_quality_gate():
    p = pathlib.Path(".pre-commit-config.yaml")
    assert p.exists()
    text = p.read_text()
    assert "repos:" in text                          # it is a pre-commit config
    assert "check_quality.py --check-only" in text   # and it runs our gate
