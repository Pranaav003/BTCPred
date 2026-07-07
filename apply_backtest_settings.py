"""
apply_backtest_settings.py — Apply backtest-validated signal settings to the DB.

Run this once after deploying to apply the parameter improvements discovered by
backtest_v2.py (2026-07-07). Settings are idempotent: running multiple times is safe.

Backtest evidence (live_training_data_deduped_enriched.csv, 4,672 markets):
  • 4/4 walk-forward folds profitable  (avg WR 70.2%, avg PnL $71/fold)
  • mispricing_threshold 0.25: Sharpe 0.53 vs 0.25 at old threshold 0.10
  • yes_cutoff 0.72: Sharpe 0.41 vs 0.32 at old 0.65
  • max_entry_yes 0.65: Sharpe 0.42 vs 0.26 at old 0.80
  • max_seconds 120: Sharpe 0.44 vs 0.33 at old 180

Usage:
  # From project root with Flask context:
  python3 apply_backtest_settings.py

  # Or from within a running Flask shell:
  from apply_backtest_settings import apply_settings
  apply_settings(app)
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Settings to apply (key → value)
# All values are strings as stored in AppSettings.
# ---------------------------------------------------------------------------
VALIDATED_SETTINGS: dict[str, str] = {
    # Mispricing gap threshold: raised from 0.10 → 0.25
    # Rationale: backtest Sharpe 0.53 at 0.25 vs 0.25 at 0.10 (best single lever)
    "mispricing_threshold": "0.2500",

    # YES entry price cap: lowered from 0.80 → 0.65
    # Rationale: backtest Sharpe 0.42 at 0.65 vs 0.26 at 0.80.
    # All live losses > $3.50 were at entry prices 0.63–0.82.
    "max_entry_price_yes": "0.6500",

    # NO entry price cap: unchanged at 0.80 (NO side already works)
    "max_entry_price_no": "0.8000",

    # YES agreement cutoff: raised from 0.65 → 0.72
    # Rationale: backtest Sharpe 0.41 at 0.72 vs 0.32 at 0.65
    "profile_override_moderate_yes_cutoff": "0.7200",
    "profile_override_moderate_no_cutoff":  "0.2800",

    # Time window: tightened from 60-300s → 60-120s
    # Rationale: backtest Sharpe 0.44 at 120s vs 0.33 at 180s; 300s was worst
    "profile_override_moderate_min_seconds": "60",
    "profile_override_moderate_max_seconds": "120",

    # Signal mode: ensure ensemble is active (it combines agreement + mispricing)
    "signal_mode": "ensemble",
}


def apply_settings(app=None) -> None:
    """Apply validated settings via app context. Prints each change."""
    if app is None:
        # Import flask app when running as standalone script
        sys.path.insert(0, str(Path(__file__).parent))
        from app import create_app
        app = create_app()

    with app.app_context():
        from app.db_helpers import get_setting, set_setting

        print("Applying backtest-validated settings:")
        print("-" * 60)
        for key, new_value in VALIDATED_SETTINGS.items():
            old_value = get_setting(key)
            if old_value == new_value:
                print(f"  {'(unchanged)':<14} {key} = {new_value}")
            else:
                set_setting(key, new_value)
                old_display = repr(old_value) if old_value is not None else "(not set)"
                print(f"  {'CHANGED':<14} {key}: {old_display} → {new_value!r}")
        print("-" * 60)
        print("Done. Restart the scheduler to pick up changes.")


if __name__ == "__main__":
    apply_settings()
