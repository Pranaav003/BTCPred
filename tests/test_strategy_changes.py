"""Tests for strategy logic changes."""
import pytest


class TestVolatilityGuardMispricingBypass:
    def test_mispricing_bypasses_volatility_guard(self):
        """Volatility guard only blocks agreement — mispricing signals still trade."""
        from app.signal_engine import evaluate_ensemble_signal
        result = evaluate_ensemble_signal(
            p_market=0.30, p_raw=0.60,
            seconds_to_close=120, entry_bucket=120,
            yes_cutoff=0.65, mispricing_threshold=0.20,
            min_seconds=60, max_seconds=180,
            volatility_guard_active=True,
        )
        assert result.signal == "PAPER BUY YES", "Mispricing should bypass volatility guard"
        assert result.agreement_region == "model_bullish"
