"""Tests for paper_trading: compute_position_size and position_sizing_breakdown."""
import importlib
import importlib.util
import os
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Stub out heavy dependencies so that importing app.paper_trading works
# without Flask, SQLAlchemy, etc.
# ---------------------------------------------------------------------------


def _stub(name, **attrs):
    if name not in sys.modules:
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        sys.modules[name] = mod
    else:
        # Module already registered (e.g. by a prior test); add missing attrs.
        mod = sys.modules[name]
        for k, v in attrs.items():
            if not hasattr(mod, k):
                setattr(mod, k, v)
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

# Stub models that paper_trading imports.
_mock_portfolio_cls = MagicMock()
_mock_portfolio_instance = MagicMock()
_mock_portfolio_instance.cash = 1000.0
_mock_portfolio_cls.get_or_create.return_value = _mock_portfolio_instance

_stub("app.models",
      AppSettings=type("AppSettings", (), {}),
      Market=MagicMock(),
      PaperTrade=MagicMock(),
      Portfolio=_mock_portfolio_cls,
      Signal=MagicMock(),
      TradeSnapshot=MagicMock(),
      db=MagicMock())
_stub("app.db_helpers", get_or_create_market=MagicMock(), get_setting=lambda k, d=None: d, set_setting=lambda k, v: None)
_stub("app.feature_engineering", get_live_snapshot=lambda: None)
_stub("app.kalshi_client", get_active_market=lambda: None)
_stub("app.model_loader", predict_proba_raw=lambda _: 0.5)
_stub("app.routes", __path__=[])
_stub("app.routes.api", api_bp=type("bp", (), {}))
_stub("app.routes.dashboard", dashboard_bp=type("bp", (), {}))
_stub("app.scheduler", init_scheduler=lambda app: None)

# Import signal_engine first since paper_trading imports from it.
_se_path = os.path.join(os.path.dirname(__file__), "..", "app", "signal_engine.py")
_se_path = os.path.abspath(_se_path)
_se_spec = importlib.util.spec_from_file_location("app.signal_engine", _se_path)
_se_mod = importlib.util.module_from_spec(_se_spec)
sys.modules["app.signal_engine"] = _se_mod
_se_spec.loader.exec_module(_se_mod)

# Now import paper_trading via importlib.
_pt_path = os.path.join(os.path.dirname(__file__), "..", "app", "paper_trading.py")
_pt_path = os.path.abspath(_pt_path)
_pt_spec = importlib.util.spec_from_file_location("app.paper_trading", _pt_path)
_pt_mod = importlib.util.module_from_spec(_pt_spec)
sys.modules["app.paper_trading"] = _pt_mod
_pt_spec.loader.exec_module(_pt_mod)

compute_position_size = _pt_mod.compute_position_size
position_sizing_breakdown = _pt_mod.position_sizing_breakdown


# ---------------------------------------------------------------------------
# compute_position_size tests
# ---------------------------------------------------------------------------


class TestComputePositionSize:
    """Tests for compute_position_size edge-based scaling."""

    def setup_method(self):
        """Reset mock portfolio before each test."""
        _mock_portfolio_instance.cash = 1000.0

    def test_high_edge_yes_side(self):
        """YES side with p_market=0.50 -> edge=0.50 >= 0.35 -> 1.5x multiplier."""
        # base=10, edge=0.50, base_mult=1.5, no mispricing mult
        # final_mult = min(1.5*1.0, 3.0) = 1.5
        # scaled = 10 * 1.5 = 15.0, max = 1000*0.4 = 400.0
        # final = min(15.0, 400.0) = 15.0
        result = compute_position_size(
            p_market=0.50,
            side="YES",
            base_size=10.0,
        )
        assert result == 15.0

    def test_medium_edge_yes_side(self):
        """YES side with p_market=0.70 -> edge=0.30 -> 1.0x multiplier."""
        result = compute_position_size(
            p_market=0.70,
            side="YES",
            base_size=10.0,
        )
        assert result == 10.0

    def test_low_edge_yes_side(self):
        """YES side with p_market=0.85 -> edge=0.15 -> 0.6x multiplier."""
        result = compute_position_size(
            p_market=0.85,
            side="YES",
            base_size=10.0,
        )
        assert result == 6.0

    def test_very_low_edge_yes_side(self):
        """YES side with p_market=0.93 -> edge=0.07 -> 0.3x multiplier."""
        result = compute_position_size(
            p_market=0.93,
            side="YES",
            base_size=10.0,
        )
        assert result == 3.0

    def test_no_side_uses_market_prob_as_edge(self):
        """NO side with p_market=0.50 -> edge=0.50 -> 1.5x multiplier."""
        result = compute_position_size(
            p_market=0.50,
            side="NO",
            base_size=10.0,
        )
        assert result == 15.0

    def test_no_side_low_edge(self):
        """NO side with p_market=0.10 -> edge=0.10 -> 0.6x multiplier."""
        result = compute_position_size(
            p_market=0.10,
            side="NO",
            base_size=10.0,
        )
        assert result == 6.0

    def test_zero_base_size(self):
        """Zero base_size returns 0.0."""
        result = compute_position_size(
            p_market=0.50,
            side="YES",
            base_size=0.0,
        )
        assert result == 0.0

    def test_negative_base_size(self):
        """Negative base_size returns 0.0."""
        result = compute_position_size(
            p_market=0.50,
            side="YES",
            base_size=-5.0,
        )
        assert result == 0.0

    def test_mispricing_mode_high_gap(self):
        """Mispricing mode with gap >= 0.40 -> 2.0x mispricing multiplier."""
        # edge for YES at p_market=0.50 = 0.50 -> base_mult=1.5
        # mispricing mode, gap=0.40 -> mp_mult=2.0
        # final_mult = min(1.5*2.0, 3.0) = 3.0
        # scaled = 10 * 3.0 = 30.0
        result = compute_position_size(
            p_market=0.50,
            side="YES",
            base_size=10.0,
            mispricing_gap=0.40,
            signal_mode="mispricing",
        )
        assert result == 30.0

    def test_mispricing_mode_medium_gap(self):
        """Mispricing mode with gap >= 0.30 -> 1.75x mispricing multiplier."""
        # edge=0.50 -> base_mult=1.5
        # gap=0.30 -> mp_mult=1.75
        # final_mult = min(1.5*1.75, 3.0) = min(2.625, 3.0) = 2.625
        # scaled = 10 * 2.625 = 26.25
        result = compute_position_size(
            p_market=0.50,
            side="YES",
            base_size=10.0,
            mispricing_gap=0.30,
            signal_mode="mispricing",
        )
        assert result == 26.25

    def test_mispricing_mode_low_gap(self):
        """Mispricing mode with gap >= 0.20 -> 1.5x mispricing multiplier."""
        # edge=0.50 -> base_mult=1.5
        # gap=0.20 -> mp_mult=1.5
        # final_mult = min(1.5*1.5, 3.0) = min(2.25, 3.0) = 2.25
        # scaled = 10 * 2.25 = 22.5
        result = compute_position_size(
            p_market=0.50,
            side="YES",
            base_size=10.0,
            mispricing_gap=0.20,
            signal_mode="mispricing",
        )
        assert result == 22.5

    def test_mispricing_mode_tiny_gap(self):
        """Mispricing mode with gap < 0.20 -> 1.0x mispricing multiplier."""
        # edge=0.50 -> base_mult=1.5
        # gap=0.10 -> mp_mult=1.0
        # final_mult = min(1.5*1.0, 3.0) = 1.5
        # scaled = 10 * 1.5 = 15.0
        result = compute_position_size(
            p_market=0.50,
            side="YES",
            base_size=10.0,
            mispricing_gap=0.10,
            signal_mode="mispricing",
        )
        assert result == 15.0

    def test_agreement_mode_ignores_mispricing_gap(self):
        """Agreement mode ignores mispricing_gap (mp_mult=1.0 always)."""
        result = compute_position_size(
            p_market=0.50,
            side="YES",
            base_size=10.0,
            mispricing_gap=0.40,
            signal_mode="agreement",
        )
        # base_mult=1.5, mp_mult=1.0, final=min(1.5, 3.0)=1.5
        assert result == 15.0

    def test_combined_multiplier_capped_at_3x(self):
        """Combined multiplier is capped at 3.0."""
        # edge=0.50 -> base_mult=1.5, gap=0.40 -> mp_mult=2.0
        # 1.5 * 2.0 = 3.0, which is exactly at cap
        # Try to exceed: edge >= 0.35 -> base_mult=1.5, gap >= 0.40 -> mp_mult=2.0
        # Can't exceed 3.0 with these tiers, so just verify cap holds
        result = compute_position_size(
            p_market=0.50,
            side="YES",
            base_size=100.0,
            mispricing_gap=0.40,
            signal_mode="mispricing",
        )
        # 100 * min(1.5*2.0, 3.0) = 100 * 3.0 = 300.0
        assert result == 300.0

    def test_cash_cap_at_40_percent(self):
        """Final size is capped at 40% of portfolio cash."""
        _mock_portfolio_instance.cash = 50.0
        # base=100, edge=0.50 -> mult=1.5, scaled=150.0
        # max = 50 * 0.4 = 20.0
        result = compute_position_size(
            p_market=0.50,
            side="YES",
            base_size=100.0,
        )
        assert result == 20.0

    def test_volatility_override_caps_at_1x(self):
        """Volatility override caps final_multiplier at 1.0."""
        # edge=0.50 -> base_mult=1.5, but volatility_override caps at 1.0
        # scaled = 10 * 1.0 = 10.0
        result = compute_position_size(
            p_market=0.50,
            side="YES",
            base_size=10.0,
            volatility_override=True,
        )
        assert result == 10.0

    def test_side_case_insensitive(self):
        """Side parameter is case-insensitive."""
        result_lower = compute_position_size(p_market=0.50, side="yes", base_size=10.0)
        result_upper = compute_position_size(p_market=0.50, side="YES", base_size=10.0)
        assert result_lower == result_upper


# ---------------------------------------------------------------------------
# position_sizing_breakdown tests
# ---------------------------------------------------------------------------


class TestPositionSizingBreakdown:
    """Tests for position_sizing_breakdown (no DB access)."""

    def test_high_edge_yes(self):
        """YES side with high edge -> base_mult=1.5."""
        result = position_sizing_breakdown(
            p_market=0.50,
            side="YES",
        )
        assert result["base_multiplier"] == 1.5
        assert result["mispricing_multiplier"] == 1.0
        assert result["final_multiplier"] == 1.5

    def test_medium_edge_yes(self):
        """YES side with medium edge -> base_mult=1.0."""
        result = position_sizing_breakdown(
            p_market=0.70,
            side="YES",
        )
        assert result["base_multiplier"] == 1.0

    def test_low_edge_yes(self):
        """YES side with low edge -> base_mult=0.6."""
        result = position_sizing_breakdown(
            p_market=0.85,
            side="YES",
        )
        assert result["base_multiplier"] == 0.6

    def test_very_low_edge_yes(self):
        """YES side with very low edge -> base_mult=0.3."""
        result = position_sizing_breakdown(
            p_market=0.93,
            side="YES",
        )
        assert result["base_multiplier"] == 0.3

    def test_no_side_edge(self):
        """NO side uses p_market as edge directly."""
        result = position_sizing_breakdown(
            p_market=0.50,
            side="NO",
        )
        # edge = p_market = 0.50 >= 0.35 -> base_mult=1.5
        assert result["base_multiplier"] == 1.5

    def test_mispricing_mode_high_gap(self):
        """Mispricing mode with gap >= 0.40 -> mp_mult=2.0."""
        result = position_sizing_breakdown(
            p_market=0.50,
            side="YES",
            mispricing_gap=0.40,
            signal_mode="mispricing",
        )
        assert result["mispricing_multiplier"] == 2.0
        assert result["final_multiplier"] == min(1.5 * 2.0, 3.0)

    def test_agreement_mode_ignores_gap(self):
        """Agreement mode always sets mp_mult=1.0."""
        result = position_sizing_breakdown(
            p_market=0.50,
            side="YES",
            mispricing_gap=0.50,
            signal_mode="agreement",
        )
        assert result["mispricing_multiplier"] == 1.0

    def test_final_multiplier_capped_at_3(self):
        """Combined multiplier is capped at 3.0."""
        result = position_sizing_breakdown(
            p_market=0.50,
            side="YES",
            mispricing_gap=0.40,
            signal_mode="mispricing",
        )
        assert result["final_multiplier"] <= 3.0
