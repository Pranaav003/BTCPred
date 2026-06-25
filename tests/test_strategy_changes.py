"""Tests for strategy logic changes."""
import pytest


class TestVolatilityGuardNoBypass:
    def test_mispricing_blocked_by_volatility_guard(self):
        """After the change, mispricing trades should also be blocked by volatility guard."""
        from app.signal_engine import evaluate_ensemble_signal
        result = evaluate_ensemble_signal(
            p_market=0.30, p_raw=0.60,
            seconds_to_close=120, entry_bucket=120,
            yes_cutoff=0.65, mispricing_threshold=0.20,
            min_seconds=60, max_seconds=180,
            volatility_guard_active=True,
        )
        assert result.signal == "NO SIGNAL", "Mispricing should now be blocked by volatility guard"
        assert result.agreement_region == "volatility_guard"
