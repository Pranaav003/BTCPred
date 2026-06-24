"""Tests for consistent max_daily_loss default across codebase."""


def _find_max_daily_loss_defaults():
    """Scan source files for max_daily_loss default values."""
    results = {}
    with open("app/scheduler.py") as f:
        content = f.read()
    for line in content.splitlines():
        if 'AppSettings.get("max_daily_loss"' in line:
            if '"200.0"' in line:
                results["scheduler_paper"] = "200.0"
            elif '"50.0"' in line:
                results["scheduler_paper"] = "50.0"
    with open("app/db_helpers.py") as f:
        content = f.read()
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith('"max_daily_loss"'):
            if '"200.0"' in stripped:
                results["db_helpers_default"] = "200.0"
            elif '"50.0"' in stripped:
                results["db_helpers_default"] = "50.0"
    with open("app/paper_trading.py") as f:
        content = f.read()
    for line in content.splitlines():
        if 'AppSettings.get("max_daily_loss"' in line:
            if '"200.0"' in line:
                results["paper_trading"] = "200.0"
            elif '"50.0"' in line:
                results["paper_trading"] = "50.0"
    return results


def test_max_daily_loss_defaults_are_consistent():
    """All max_daily_loss defaults should be 50.0 across the codebase."""
    defaults = _find_max_daily_loss_defaults()
    for location, value in defaults.items():
        assert value == "50.0", f"{location} has max_daily_loss default {value}, expected 50.0"
