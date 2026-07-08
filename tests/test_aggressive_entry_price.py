"""Aggressive (marketable) fill pricing (2026-07-08).

Paper/backtest shows the mispricing signal is strongly +EV (backtest
+$0.082/contract; paper 78.7% WR) but live under-realizes it: orders were
priced at the STALE candle mid + a fixed offset and rested as passive limits,
filling only ~65% of the time and adversely selecting (fill only when the
market moved against us -> live WR 59% vs paper 78%).

`_aggressive_entry_price` prices a marketable order that crosses the LIVE
orderbook ask for an immediate fill, capped by max-entry so it never chases
past the cap. Crossing-cost sweep: stays +EV up to ~11c/contract; real cost
~1-3c. This test pins the pricing behavior.
"""
import os
import sys
import types

import pytest

_REAL_APP_DIR = os.path.join(os.path.dirname(__file__), "..", "app")
_SHADOW_ROOTS = (
    "app", "flask", "flask_sqlalchemy", "flask_migrate", "flask_wtf",
    "sqlalchemy", "click", "dotenv",
)


def _is_shadow(key: str) -> bool:
    return any(key == r or key.startswith(r + ".") for r in _SHADOW_ROOTS)


@pytest.fixture
def scheduler():
    """Import the real app.scheduler (pure helper needs no DB/app context)."""
    saved = {k: sys.modules.pop(k) for k in list(sys.modules) if _is_shadow(k)}
    app_pkg = types.ModuleType("app")
    app_pkg.__path__ = [_REAL_APP_DIR]
    app_pkg.__package__ = "app"
    sys.modules["app"] = app_pkg
    try:
        import app.scheduler as scheduler
        yield scheduler
    finally:
        for k in [k for k in list(sys.modules) if _is_shadow(k)]:
            sys.modules.pop(k, None)
        sys.modules.update(saved)


def test_yes_crosses_live_yes_ask(scheduler):
    # mid would be 55c; live ask is 60c -> price off the ASK, not the mid.
    entry, cents = scheduler._aggressive_entry_price(
        "yes", p_market=0.55, quote={"yes_ask": 0.60, "no_ask": 0.42},
        max_entry_yes=0.65, max_entry_no=0.80, buffer_cents=1,
    )
    assert entry == 0.60
    assert cents == 61  # ask 60c + 1c buffer to ensure the cross


def test_no_crosses_live_no_ask(scheduler):
    # market YES 45% -> NO mid 55c; live no_ask 58c -> price off the no_ask.
    entry, cents = scheduler._aggressive_entry_price(
        "no", p_market=0.45, quote={"yes_ask": 0.60, "no_ask": 0.58},
        max_entry_yes=0.65, max_entry_no=0.80, buffer_cents=1,
    )
    assert entry == 0.58
    assert cents == 59


def test_prices_off_ask_not_mid(scheduler):
    # Regression: the old bug priced off result.p_market (mid) + fixed offset.
    entry, cents = scheduler._aggressive_entry_price(
        "yes", p_market=0.50, quote={"yes_ask": 0.62, "no_ask": 0.40},
        max_entry_yes=0.65, max_entry_no=0.80, buffer_cents=1,
    )
    assert cents == 63  # 62 (ask) + 1, NOT 51 (mid 50 + 1)


def test_falls_back_to_mid_when_no_quote(scheduler):
    entry, cents = scheduler._aggressive_entry_price(
        "yes", p_market=0.55, quote=None,
        max_entry_yes=0.65, max_entry_no=0.80, buffer_cents=1,
    )
    assert entry == 0.55
    assert cents == 56


def test_skips_when_ask_exceeds_yes_cap(scheduler):
    result = scheduler._aggressive_entry_price(
        "yes", p_market=0.55, quote={"yes_ask": 0.70, "no_ask": 0.30},
        max_entry_yes=0.65, max_entry_no=0.80, buffer_cents=1,
    )
    assert result is None  # never chase past the max-entry cap


def test_skips_when_no_ask_exceeds_no_cap(scheduler):
    result = scheduler._aggressive_entry_price(
        "no", p_market=0.10, quote={"yes_ask": 0.12, "no_ask": 0.90},
        max_entry_yes=0.65, max_entry_no=0.80, buffer_cents=1,
    )
    assert result is None
