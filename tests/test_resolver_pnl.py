"""Tests for PnL calculation in _apply_kalshi_settlement."""
import importlib
import importlib.util
import os
import sys
import types
import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Stub out ALL heavy dependencies so that importing app.resolver works
# without Flask, SQLAlchemy, etc.
# ---------------------------------------------------------------------------

# Helper: create a stub module and register it.
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

# Now stub everything the submodules might import.
_stub("flask", Flask=type("Flask", (), {}), Blueprint=type("Blueprint", (), {}))
_stub("flask_sqlalchemy", SQLAlchemy=type("SQLAlchemy", (), {"init_app": lambda self, app: None}))
_stub("sqlalchemy", text=lambda s: s)
_stub("dotenv")
_stub("click")
_stub("app.config", config_by_name={"development": type("C", (), {})()})
_stub("app.extensions", db=MagicMock())
_stub("app.db_helpers", resolve_paper_trades=MagicMock())
_stub("app.kalshi_client", get_market_resolution=MagicMock())
_stub("app.models", Market=MagicMock(), Signal=MagicMock(), AppSettings=MagicMock())

# Provide the _parse_fp_count / _parse_fp_dollars that resolver imports lazily.
def _parse_fp_count(val):
    if val is None:
        return 0.0
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def _parse_fp_dollars(val):
    if val is None:
        return 0.0
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


_stub("app.kalshi_trader", _parse_fp_count=_parse_fp_count, _parse_fp_dollars=_parse_fp_dollars)

# Stub additional submodules that __init__.py or other modules may import.
_stub("app.model_loader", get_model=MagicMock())
_stub("app.routes", __path__=[])
_stub("app.routes.api", api_bp=MagicMock())
_stub("app.routes.dashboard", dashboard_bp=MagicMock())
_stub("app.scheduler", init_scheduler=MagicMock())

# Now import resolver via importlib so it picks up the stubbed 'app' package.
_resolver_path = os.path.join(os.path.dirname(__file__), "..", "app", "resolver.py")
_resolver_path = os.path.abspath(_resolver_path)
_spec = importlib.util.spec_from_file_location("app.resolver", _resolver_path)
_resolver_mod = importlib.util.module_from_spec(_spec)
sys.modules["app.resolver"] = _resolver_mod
_spec.loader.exec_module(_resolver_mod)

_apply_kalshi_settlement = _resolver_mod._apply_kalshi_settlement


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def make_trade():
    """Factory for LiveTrade-like objects with configurable side."""
    def _make(side="YES", contracts=10, entry_price=0.70, cost_dollars=7.00):
        trade = MagicMock()
        trade.side = side
        trade.contracts = contracts
        trade.entry_price = entry_price
        trade.cost_dollars = cost_dollars
        return trade
    return _make


def _make_settlement(market_result="yes", yes_count=10, no_count=0,
                     yes_cost=7.00, no_cost=0.0, fee_cost=0.07,
                     revenue=1000):
    """Build a settlement dict matching Kalshi API shape."""
    return {
        "market_result": market_result,
        "yes_count_fp": f"{float(yes_count):.0f}",
        "no_count_fp": f"{float(no_count):.0f}",
        "yes_total_cost_dollars": f"{yes_cost:.2f}",
        "no_total_cost_dollars": f"{no_cost:.2f}",
        "fee_cost": f"{fee_cost:.2f}",
        "revenue": revenue,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_yes_trade_correct_pnl_uses_side_cost_only(make_trade):
    """PnL for a winning YES trade must use yes_cost, not yes_cost+no_cost."""
    trade = make_trade(side="YES", contracts=10)
    settlement = _make_settlement(
        market_result="yes",
        yes_count=10,
        yes_cost=7.00,
        no_cost=0.00,
        fee_cost=0.07,
        revenue=1000,
    )
    now = datetime.now(timezone.utc)
    _apply_kalshi_settlement(trade, settlement, now)
    assert trade.realized_pnl == 2.93


def test_yes_trade_correct_pnl_with_nonzero_no_cost(make_trade):
    """Even if no_cost is non-zero, YES PnL must not subtract it."""
    trade = make_trade(side="YES", contracts=10)
    settlement = _make_settlement(
        market_result="yes",
        yes_count=10,
        yes_cost=7.00,
        no_cost=3.00,
        fee_cost=0.07,
        revenue=1000,
    )
    now = datetime.now(timezone.utc)
    _apply_kalshi_settlement(trade, settlement, now)
    assert trade.realized_pnl == 2.93


def test_no_trade_correct_pnl_uses_no_cost(make_trade):
    """PnL for a winning NO trade must use no_cost, not yes_cost+no_cost."""
    trade = make_trade(side="NO", contracts=10)
    settlement = _make_settlement(
        market_result="no",
        yes_count=0,
        no_count=10,
        yes_cost=0.00,
        no_cost=3.00,
        fee_cost=0.03,
        revenue=1000,
    )
    now = datetime.now(timezone.utc)
    _apply_kalshi_settlement(trade, settlement, now)
    assert trade.realized_pnl == 6.97


def test_yes_trade_wrong_pnl(make_trade):
    """Losing YES trade: PnL = revenue/100 - side_cost - fee (negative)."""
    trade = make_trade(side="YES", contracts=10)
    settlement = _make_settlement(
        market_result="no",
        yes_count=10,
        yes_cost=7.00,
        no_cost=0.00,
        fee_cost=0.07,
        revenue=0,
    )
    now = datetime.now(timezone.utc)
    _apply_kalshi_settlement(trade, settlement, now)
    assert trade.realized_pnl == -7.07
