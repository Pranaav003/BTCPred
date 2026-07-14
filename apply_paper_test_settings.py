"""
apply_paper_test_settings.py — Turn on PAPER-mode forward-testing of the
backtest-validated ensemble config (NO real money).

Purpose
-------
The sim/ strategy-search sweep (2026-07-14) confirmed the ensemble signal has a
statistically significant edge (MC p≈0, 4/4 walk-forward folds), but its best
out-of-sample estimate rested on only ~23 test trades. This script enables PAPER
auto-trading on the config that is (a) coherent with the app's profile system and
(b) backed by the larger 4,672-market backtest, so the strategy accumulates real
forward out-of-sample trades before any live-money change.

It reuses the signal parameters from apply_backtest_settings.VALIDATED_SETTINGS
(mispricing_threshold 0.25, moderate profile 60-120s, yes_cutoff 0.72,
no_max_p_raw 0.20) and layers on the paper-trading enable flags.

Notes / honest caveats
----------------------
- Exits: the search found stop-loss / take-profit / trailing did NOT beat
  hold-to-resolution. The app already holds to resolution, so there is nothing to
  change and no live-exit code is needed.
- Sizing: the search's winning "payoff_aware" sizing is a backtest-only construct
  with no live/paper equivalent. Paper uses the app's existing compute_position_size.
  This forward-test validates the ENTRY EDGE, not sizing.
- Threshold: this uses 0.25 (larger-backtest choice), NOT the sweep's 0.15 (which
  rested on the thin 23-trade holdout). Re-evaluate 0.15 once forward data grows.

Usage
-----
  # From project root with Flask context (run where the app's DB lives):
  python3 apply_paper_test_settings.py

  # Or from within a running Flask shell:
  from apply_paper_test_settings import apply_paper_test
  apply_paper_test(app)

Safety: this sets live_trading_enabled="false" explicitly. It only ever enables
PAPER trading. Restart the scheduler afterward to pick up the changes.
"""

from __future__ import annotations

import sys
from pathlib import Path

from apply_backtest_settings import VALIDATED_SETTINGS

# Paper-trading enable flags layered on top of the validated signal config.
PAPER_ENABLE_SETTINGS: dict[str, str] = {
    # --- Turn ON paper trading (simulated; NO real money) ---
    "paper_trading_enabled": "true",
    "auto_trade_enabled": "true",
    "scheduler_running": "true",
    "paper_trade_size": "10.0",
    "dynamic_sizing_enabled": "false",  # clean flat-ish sizing for an edge-focused test

    # --- Safety: keep LIVE (real-money) trading OFF ---
    "live_trading_enabled": "false",
}

# Full set applied by this script: validated signal params + paper enable flags.
PAPER_TEST_SETTINGS: dict[str, str] = {**VALIDATED_SETTINGS, **PAPER_ENABLE_SETTINGS}


def apply_paper_test(app=None) -> None:
    """Apply paper-test settings via app context. Idempotent; prints each change."""
    if app is None:
        sys.path.insert(0, str(Path(__file__).parent))
        from app import create_app
        app = create_app()

    with app.app_context():
        from app.db_helpers import get_setting, set_setting

        print("Applying PAPER-mode forward-test settings (no real money):")
        print("-" * 64)
        for key, new_value in PAPER_TEST_SETTINGS.items():
            old_value = get_setting(key)
            if old_value == new_value:
                print(f"  {'(unchanged)':<14} {key} = {new_value}")
            else:
                set_setting(key, new_value)
                old_display = repr(old_value) if old_value is not None else "(not set)"
                print(f"  {'CHANGED':<14} {key}: {old_display} → {new_value!r}")
        print("-" * 64)

        # Explicit safety assertion: never leave live trading on.
        assert get_setting("live_trading_enabled") == "false", \
            "SAFETY: live_trading_enabled must be 'false' after paper-test setup"
        print("Live trading confirmed OFF. Paper trading ON.")
        print("Done. Restart the scheduler to pick up changes.")


if __name__ == "__main__":
    apply_paper_test()
