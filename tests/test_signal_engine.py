"""Tests for signal_engine: agreement region, agreement signals, mispricing, and ensemble."""
import importlib
import importlib.util
import os
import sys
import types

import pytest

# ---------------------------------------------------------------------------
# Stub out heavy dependencies so that importing app.signal_engine works
# without Flask, SQLAlchemy, model files, etc.
# ---------------------------------------------------------------------------


def _stub(name, **attrs):
    if name not in sys.modules:
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        sys.modules[name] = mod
    return sys.modules[name]


# Stub the top-level 'app' package so its __init__.py does NOT run.
_app_stub = types.ModuleType("app")
_app_stub.__path__ = [os.path.join(os.path.dirname(__file__), "..", "app")]
_app_stub.__package__ = "app"
sys.modules["app"] = _app_stub

_stub("flask", Flask=type("Flask", (), {}))
_stub("flask_sqlalchemy", SQLAlchemy=type("SQLAlchemy", (), {}))
_stub("flask_migrate", Migrate=type("Migrate", (), {}))
_stub("flask_wtf")
_stub("flask_wtf.csrf", CSRFProtect=type("CSRFProtect", (), {}))
_stub("sqlalchemy", func=type("func", (), {}))
_stub("dotenv")
_stub("click")
_stub("app.config", config_by_name={"development": type("C", (), {})()})
_stub("app.extensions", db=type("db", (), {}))
_stub("app.model_loader", predict_proba_raw=lambda _: 0.5)
_stub("app.models", AppSettings=type("AppSettings", (), {}))
_stub("app.db_helpers")
_stub("app.kalshi_client")
_stub("app.routes", __path__=[])
_stub("app.routes.api", api_bp=type("bp", (), {}))
_stub("app.routes.dashboard", dashboard_bp=type("bp", (), {}))
_stub("app.scheduler", init_scheduler=lambda app: None)

# Now import signal_engine via importlib.
_se_path = os.path.join(os.path.dirname(__file__), "..", "app", "signal_engine.py")
_se_path = os.path.abspath(_se_path)
_spec = importlib.util.spec_from_file_location("app.signal_engine", _se_path)
_se_mod = importlib.util.module_from_spec(_spec)
sys.modules["app.signal_engine"] = _se_mod
_spec.loader.exec_module(_se_mod)

determine_agreement_region = _se_mod.determine_agreement_region
evaluate_signal = _se_mod.evaluate_signal
evaluate_mispricing_signal = _se_mod.evaluate_mispricing_signal
evaluate_ensemble_signal = _se_mod.evaluate_ensemble_signal
SignalResult = _se_mod.SignalResult
MIN_ENTRY_PRICE = _se_mod.MIN_ENTRY_PRICE


# ---------------------------------------------------------------------------
# determine_agreement_region tests
# ---------------------------------------------------------------------------


class TestDetermineAgreementRegion:
    """Tests for determine_agreement_region."""

    def test_agree_yes(self):
        """Both p_market and p_raw above yes_cutoff -> agree_yes."""
        assert determine_agreement_region(0.75, 0.80, 0.65, 0.35) == "agree_yes"

    def test_agree_yes_at_cutoff(self):
        """Exactly at cutoff counts as agreement."""
        assert determine_agreement_region(0.65, 0.65, 0.65, 0.35) == "agree_yes"

    def test_agree_no(self):
        """Both p_market and p_raw below no_cutoff -> agree_no."""
        assert determine_agreement_region(0.20, 0.25, 0.65, 0.35) == "agree_no"

    def test_agree_no_at_cutoff(self):
        """Exactly at no_cutoff counts as agreement."""
        assert determine_agreement_region(0.35, 0.30, 0.65, 0.35) == "agree_no"

    def test_market_yes_raw_no(self):
        """Market >= 0.5 but raw < 0.5 -> market_yes_raw_no."""
        assert determine_agreement_region(0.60, 0.40, 0.65, 0.35) == "market_yes_raw_no"

    def test_market_no_raw_yes(self):
        """Market < 0.5 but raw >= 0.5 -> market_no_raw_yes."""
        assert determine_agreement_region(0.40, 0.60, 0.65, 0.35) == "market_no_raw_yes"

    def test_no_agreement(self):
        """Both above 0.5 but neither reaches yes_cutoff -> no_agreement."""
        assert determine_agreement_region(0.55, 0.55, 0.65, 0.35) == "no_agreement"

    def test_no_agreement_both_below_half(self):
        """Both below 0.5 but neither at no_cutoff -> no_agreement."""
        assert determine_agreement_region(0.45, 0.40, 0.65, 0.35) == "no_agreement"


# ---------------------------------------------------------------------------
# evaluate_signal tests
# ---------------------------------------------------------------------------


class TestEvaluateSignal:
    """Tests for evaluate_signal (agreement mode)."""

    def test_agree_yes_produces_buy_yes(self):
        """Strong YES agreement within time window -> PAPER BUY YES."""
        result = evaluate_signal(
            p_market=0.75,
            p_raw=0.80,
            seconds_to_close=120,
            entry_bucket=120,
            yes_cutoff=0.65,
            no_cutoff=0.35,
            min_seconds=60,
            max_seconds=180,
        )
        assert result.signal == "PAPER BUY YES"
        assert result.agreement_region == "agree_yes"

    def test_outside_time_window_no_signal(self):
        """Seconds_to_close outside window -> NO SIGNAL."""
        result = evaluate_signal(
            p_market=0.75,
            p_raw=0.80,
            seconds_to_close=500,
            entry_bucket=300,
            yes_cutoff=0.65,
            no_cutoff=0.35,
            min_seconds=60,
            max_seconds=180,
        )
        assert result.signal == "NO SIGNAL"
        assert result.agreement_region == "outside_time_window"

    def test_too_early_no_signal(self):
        """More than 600s to close and not forced -> NO SIGNAL."""
        result = evaluate_signal(
            p_market=0.75,
            p_raw=0.80,
            seconds_to_close=700,
            entry_bucket=300,
            yes_cutoff=0.65,
            no_cutoff=0.35,
            min_seconds=60,
            max_seconds=180,
        )
        assert result.signal == "NO SIGNAL"
        assert result.agreement_region == "outside_time_window"

    def test_agree_no_disabled_by_default(self):
        """agreement NO with enable_no_signals=False -> NO SIGNAL."""
        result = evaluate_signal(
            p_market=0.20,
            p_raw=0.25,
            seconds_to_close=90,
            entry_bucket=120,
            yes_cutoff=0.65,
            no_cutoff=0.35,
            min_seconds=60,
            max_seconds=180,
            enable_no_signals=False,
        )
        assert result.signal == "NO SIGNAL"

    def test_agree_no_enabled_produces_buy_no(self):
        """agreement NO with enable_no_signals=True and within 120s -> PAPER BUY NO."""
        result = evaluate_signal(
            p_market=0.20,
            p_raw=0.25,
            seconds_to_close=90,
            entry_bucket=120,
            yes_cutoff=0.65,
            no_cutoff=0.35,
            min_seconds=60,
            max_seconds=180,
            enable_no_signals=True,
        )
        assert result.signal == "PAPER BUY NO"
        assert result.agreement_region == "agree_no"

    def test_no_signal_suppressed_beyond_120s(self):
        """NO signals are suppressed when seconds_to_close > 120."""
        result = evaluate_signal(
            p_market=0.20,
            p_raw=0.25,
            seconds_to_close=150,
            entry_bucket=180,
            yes_cutoff=0.65,
            no_cutoff=0.35,
            min_seconds=60,
            max_seconds=180,
            enable_no_signals=True,
        )
        assert result.signal == "NO SIGNAL"

    def test_yes_entry_filtered_min_price(self):
        """YES signal with p_market below MIN_ENTRY_PRICE -> entry filtered."""
        # Use a yes_cutoff of 0.03 so that p_market=0.03, p_raw=0.70 both pass it
        # and produce agree_yes, then entry filter catches p_market < MIN_ENTRY_PRICE.
        result = evaluate_signal(
            p_market=0.03,
            p_raw=0.70,
            seconds_to_close=120,
            entry_bucket=120,
            yes_cutoff=0.03,
            no_cutoff=0.35,
            min_seconds=60,
            max_seconds=180,
        )
        assert result.signal == "NO SIGNAL"
        assert result.entry_filtered is True

    def test_yes_entry_filtered_max_price(self):
        """YES signal with p_market above max_entry_price_yes -> entry filtered."""
        result = evaluate_signal(
            p_market=0.90,
            p_raw=0.92,
            seconds_to_close=120,
            entry_bucket=120,
            yes_cutoff=0.65,
            no_cutoff=0.35,
            min_seconds=60,
            max_seconds=180,
            max_entry_price_yes=0.85,
        )
        assert result.signal == "NO SIGNAL"
        assert result.entry_filtered is True

    def test_no_entry_filtered_min_price(self):
        """NO signal where NO price (1 - p_market) is below MIN_ENTRY_PRICE."""
        # For agree_no: p_market <= no_cutoff AND p_raw <= no_cutoff.
        # For NO price filter: 1 - p_market < MIN_ENTRY_PRICE -> p_market > 0.95.
        # Both conditions: p_market in (0.95, no_cutoff]. Use no_cutoff=0.98, p_market=0.97.
        # But p_market=0.97 >= yes_cutoff=0.65 -> agree_yes wins first.
        # So use yes_cutoff=0.99 to make agree_yes fail (0.97 < 0.99).
        result = evaluate_signal(
            p_market=0.97,
            p_raw=0.97,
            seconds_to_close=90,
            entry_bucket=120,
            yes_cutoff=0.99,
            no_cutoff=0.98,
            min_seconds=60,
            max_seconds=180,
            enable_no_signals=True,
        )
        assert result.signal == "NO SIGNAL"
        assert result.entry_filtered is True

    def test_early_entry_window(self):
        """Early entry window with higher cutoff works when enabled."""
        result = evaluate_signal(
            p_market=0.85,
            p_raw=0.88,
            seconds_to_close=300,
            entry_bucket=300,
            yes_cutoff=0.65,
            no_cutoff=0.35,
            min_seconds=60,
            max_seconds=180,
            early_entry_enabled=True,
            early_entry_min_seconds=240,
            early_entry_max_seconds=480,
            early_entry_cutoff=0.82,
        )
        assert result.signal == "PAPER BUY YES"
        assert result.agreement_region == "agree_yes"

    def test_early_entry_below_cutoff_no_signal(self):
        """Early entry window but below the early_entry_cutoff -> NO SIGNAL."""
        result = evaluate_signal(
            p_market=0.70,
            p_raw=0.75,
            seconds_to_close=300,
            entry_bucket=300,
            yes_cutoff=0.65,
            no_cutoff=0.35,
            min_seconds=60,
            max_seconds=180,
            early_entry_enabled=True,
            early_entry_min_seconds=240,
            early_entry_max_seconds=480,
            early_entry_cutoff=0.82,
        )
        assert result.signal == "NO SIGNAL"

    def test_volatility_guard_blocks_agreement(self):
        """Volatility guard active blocks agreement trades."""
        result = evaluate_signal(
            p_market=0.75,
            p_raw=0.80,
            seconds_to_close=120,
            entry_bucket=120,
            yes_cutoff=0.65,
            no_cutoff=0.35,
            min_seconds=60,
            max_seconds=180,
            volatility_guard_active=True,
        )
        assert result.signal == "NO SIGNAL"
        assert result.agreement_region == "volatility_guard"


# ---------------------------------------------------------------------------
# evaluate_mispricing_signal tests
# ---------------------------------------------------------------------------


class TestEvaluateMispricingSignal:
    """Tests for evaluate_mispricing_signal."""

    def test_bullish_mispricing(self):
        """Model significantly above market -> PAPER BUY YES."""
        result = evaluate_mispricing_signal(
            p_market=0.50,
            p_raw=0.70,
            seconds_to_close=120,
            entry_bucket=120,
            min_seconds=60,
            max_seconds=180,
            mispricing_threshold=0.10,
        )
        assert result.signal == "PAPER BUY YES"
        assert result.agreement_region == "model_bullish"

    def test_bearish_mispricing(self):
        """Market significantly above model -> PAPER BUY NO."""
        result = evaluate_mispricing_signal(
            p_market=0.80,
            p_raw=0.50,
            seconds_to_close=120,
            entry_bucket=120,
            min_seconds=60,
            max_seconds=180,
            mispricing_threshold=0.10,
        )
        assert result.signal == "PAPER BUY NO"
        assert result.agreement_region == "model_bearish"

    def test_gap_below_threshold_no_signal(self):
        """Gap between model and market too small -> NO SIGNAL."""
        result = evaluate_mispricing_signal(
            p_market=0.50,
            p_raw=0.55,
            seconds_to_close=120,
            entry_bucket=120,
            min_seconds=60,
            max_seconds=180,
            mispricing_threshold=0.10,
        )
        assert result.signal == "NO SIGNAL"
        assert result.agreement_region == "no_agreement"

    def test_outside_window_no_signal(self):
        """Outside time window -> NO SIGNAL."""
        result = evaluate_mispricing_signal(
            p_market=0.50,
            p_raw=0.70,
            seconds_to_close=500,
            entry_bucket=300,
            min_seconds=60,
            max_seconds=180,
            mispricing_threshold=0.10,
        )
        assert result.signal == "NO SIGNAL"
        assert result.agreement_region == "outside_time_window"

    def test_yes_entry_filtered_min_price(self):
        """Bullish mispricing but YES price below minimum -> NO SIGNAL."""
        result = evaluate_mispricing_signal(
            p_market=0.03,
            p_raw=0.30,
            seconds_to_close=120,
            entry_bucket=120,
            min_seconds=60,
            max_seconds=180,
            mispricing_threshold=0.10,
        )
        assert result.signal == "NO SIGNAL"
        assert result.entry_filtered is True

    def test_yes_entry_filtered_max_price(self):
        """Bullish mispricing but YES price above max -> NO SIGNAL."""
        # gap = 0.99 - 0.50 = 0.49 >= 0.10, so mispricing fires.
        # Then p_market=0.50 <= max_entry=0.85 -> passes. Need p_market > 0.85.
        # Use p_market=0.90, p_raw=0.99, gap=0.09 < 0.10 -> won't fire.
        # Use threshold=0.05 so gap=0.09 >= 0.05 fires, then p_market=0.90 > 0.85.
        result = evaluate_mispricing_signal(
            p_market=0.90,
            p_raw=0.99,
            seconds_to_close=120,
            entry_bucket=120,
            min_seconds=60,
            max_seconds=180,
            mispricing_threshold=0.05,
            max_entry_price_yes=0.85,
        )
        assert result.signal == "NO SIGNAL"
        assert result.entry_filtered is True

    def test_no_entry_filtered_min_price(self):
        """Bearish mispricing but NO price below minimum -> NO SIGNAL."""
        result = evaluate_mispricing_signal(
            p_market=0.98,
            p_raw=0.70,
            seconds_to_close=120,
            entry_bucket=120,
            min_seconds=60,
            max_seconds=180,
            mispricing_threshold=0.10,
        )
        assert result.signal == "NO SIGNAL"
        assert result.entry_filtered is True

    def test_early_entry_uses_1_5x_threshold(self):
        """Early entry window uses 1.5x threshold."""
        # gap = 0.70 - 0.50 = 0.20, threshold=0.10, early threshold=0.15
        # 0.20 >= 0.15, so should still signal
        result = evaluate_mispricing_signal(
            p_market=0.50,
            p_raw=0.70,
            seconds_to_close=300,
            entry_bucket=300,
            min_seconds=60,
            max_seconds=180,
            mispricing_threshold=0.10,
            early_entry_enabled=True,
            early_entry_min_seconds=240,
            early_entry_max_seconds=480,
        )
        assert result.signal == "PAPER BUY YES"

    def test_early_entry_below_1_5x_threshold_no_signal(self):
        """Early entry with gap between threshold and 1.5x threshold -> NO SIGNAL."""
        # gap = 0.50 - 0.36 = 0.14, threshold=0.10, early threshold=0.15
        # 0.14 < 0.15 -> NO SIGNAL
        result = evaluate_mispricing_signal(
            p_market=0.36,
            p_raw=0.50,
            seconds_to_close=300,
            entry_bucket=300,
            min_seconds=60,
            max_seconds=180,
            mispricing_threshold=0.10,
            early_entry_enabled=True,
            early_entry_min_seconds=240,
            early_entry_max_seconds=480,
        )
        assert result.signal == "NO SIGNAL"


# ---------------------------------------------------------------------------
# evaluate_ensemble_signal tests
# ---------------------------------------------------------------------------


class TestEvaluateEnsembleSignal:
    """Tests for evaluate_ensemble_signal: agreement takes priority over mispricing."""

    def test_pure_agreement_yes(self):
        """Agreement YES without mispricing -> agreement signal."""
        result = evaluate_ensemble_signal(
            p_market=0.70,
            p_raw=0.75,
            seconds_to_close=120,
            entry_bucket=120,
            yes_cutoff=0.65,
            mispricing_threshold=0.20,
            min_seconds=60,
            max_seconds=180,
        )
        assert result.signal == "PAPER BUY YES"
        assert result.agreement_region == "agree_yes"
        assert "Agreement" in result.reason

    def test_mispricing_bullish_without_agreement(self):
        """Mispricing bullish when no agreement -> mispricing signal."""
        # p_market=0.45 (below 0.65 cutoff), p_raw=0.70 (above 0.5)
        # gap = 0.70 - 0.45 = 0.25 >= 0.20 threshold, p_raw >= 0.50
        result = evaluate_ensemble_signal(
            p_market=0.45,
            p_raw=0.70,
            seconds_to_close=120,
            entry_bucket=120,
            yes_cutoff=0.65,
            mispricing_threshold=0.20,
            min_seconds=60,
            max_seconds=180,
        )
        assert result.signal == "PAPER BUY YES"
        assert result.agreement_region == "model_bullish"

    def test_mispricing_bearish(self):
        """Bearish mispricing -> PAPER BUY NO."""
        # p_market=0.80, p_raw=0.45, gap = 0.45 - 0.80 = -0.35
        # -gap = 0.35 >= 0.20 threshold, p_raw=0.45 < 0.50
        result = evaluate_ensemble_signal(
            p_market=0.80,
            p_raw=0.45,
            seconds_to_close=120,
            entry_bucket=120,
            yes_cutoff=0.65,
            mispricing_threshold=0.20,
            min_seconds=60,
            max_seconds=180,
        )
        assert result.signal == "PAPER BUY NO"
        assert result.agreement_region == "model_bearish"

    def test_bullish_gap_but_model_bearish_skipped(self):
        """Bullish gap detected but model < 0.50 -> skip YES mispricing."""
        # p_market=0.25, p_raw=0.45, gap=0.20 >= 0.20 but p_raw=0.45 < 0.50 -> skip
        result = evaluate_ensemble_signal(
            p_market=0.25,
            p_raw=0.45,
            seconds_to_close=120,
            entry_bucket=120,
            yes_cutoff=0.65,
            mispricing_threshold=0.20,
            min_seconds=60,
            max_seconds=180,
        )
        assert result.signal == "NO SIGNAL"

    def test_bearish_gap_but_model_bullish_skipped(self):
        """Bearish gap detected but model >= 0.50 -> skip NO mispricing."""
        # p_market=0.80, p_raw=0.55, gap = -0.25, -gap = 0.25 >= 0.20
        # but p_raw=0.55 >= 0.50 -> skip
        result = evaluate_ensemble_signal(
            p_market=0.80,
            p_raw=0.55,
            seconds_to_close=120,
            entry_bucket=120,
            yes_cutoff=0.65,
            mispricing_threshold=0.20,
            min_seconds=60,
            max_seconds=180,
        )
        assert result.signal == "NO SIGNAL"

    def test_outside_window_no_signal(self):
        """Outside time window -> NO SIGNAL."""
        result = evaluate_ensemble_signal(
            p_market=0.75,
            p_raw=0.80,
            seconds_to_close=500,
            entry_bucket=300,
            yes_cutoff=0.65,
            mispricing_threshold=0.20,
            min_seconds=60,
            max_seconds=180,
        )
        assert result.signal == "NO SIGNAL"
        assert result.agreement_region == "outside_time_window"

    def test_early_entry_window(self):
        """Early entry window with effective_cutoff=early_entry_cutoff."""
        # p_market=0.83 must be <= max_entry_yes=0.85 and >= early_entry_cutoff=0.82
        result = evaluate_ensemble_signal(
            p_market=0.83,
            p_raw=0.88,
            seconds_to_close=350,
            entry_bucket=300,
            yes_cutoff=0.65,
            max_entry_yes=0.85,
            mispricing_threshold=0.20,
            min_seconds=60,
            max_seconds=180,
            early_entry_enabled=True,
            early_entry_min=300,
            early_entry_max=600,
            early_entry_cutoff=0.82,
        )
        assert result.signal == "PAPER BUY YES"
        assert result.agreement_region == "agree_yes"

    def test_volatility_guard_blocks_agreement_only(self):
        """Volatility guard blocks agreement trades."""
        result = evaluate_ensemble_signal(
            p_market=0.70,
            p_raw=0.75,
            seconds_to_close=120,
            entry_bucket=120,
            yes_cutoff=0.65,
            mispricing_threshold=0.20,
            min_seconds=60,
            max_seconds=180,
            volatility_guard_active=True,
        )
        assert result.signal == "NO SIGNAL"
        assert result.agreement_region == "volatility_guard"

    def test_volatility_guard_allows_mispricing(self):
        """Volatility guard allows mispricing trades through."""
        result = evaluate_ensemble_signal(
            p_market=0.45,
            p_raw=0.70,
            seconds_to_close=120,
            entry_bucket=120,
            yes_cutoff=0.65,
            mispricing_threshold=0.20,
            min_seconds=60,
            max_seconds=180,
            volatility_guard_active=True,
        )
        assert result.signal == "PAPER BUY YES"
        assert result.agreement_region == "model_bullish"

    def test_entry_filtered_yes_price_too_high(self):
        """YES agreement but price above max_entry_yes -> NO SIGNAL."""
        result = evaluate_ensemble_signal(
            p_market=0.90,
            p_raw=0.92,
            seconds_to_close=120,
            entry_bucket=120,
            yes_cutoff=0.65,
            max_entry_yes=0.85,
            mispricing_threshold=0.20,
            min_seconds=60,
            max_seconds=180,
        )
        assert result.signal == "NO SIGNAL"
        assert result.agreement_region == "entry_filtered"

    def test_no_agreement_no_mispricing(self):
        """No agreement and no mispricing -> NO SIGNAL."""
        result = evaluate_ensemble_signal(
            p_market=0.55,
            p_raw=0.60,
            seconds_to_close=120,
            entry_bucket=120,
            yes_cutoff=0.65,
            mispricing_threshold=0.20,
            min_seconds=60,
            max_seconds=180,
        )
        assert result.signal == "NO SIGNAL"
