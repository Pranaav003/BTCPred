"""Stale-GTC-order hygiene (2026-07-08).

A resting GTC order is placed on a mispricing signal. If the market then moves
so the live model no longer supports that side (signal collapses to NO SIGNAL
or flips), a still-resting bid would only fill on a move against us — adverse
selection. `_cancel_stale_resting_order` cancels it on the next poll instead of
waiting for the expiration.

Backtest (backtest_current_rule.py) showed the mispricing rule is +EV per
contract; the live gap vs backtest is largely fill quality / adverse selection,
which this targets.
"""
import os
import sys
import types
from datetime import datetime, timezone

import pytest

_REAL_APP_DIR = os.path.join(os.path.dirname(__file__), "..", "app")


_SHADOW_ROOTS = (
    "app", "flask", "flask_sqlalchemy", "flask_migrate", "flask_wtf",
    "sqlalchemy", "click", "dotenv",
)


def _is_shadow(key: str) -> bool:
    # Evict fakes other test files install (flask/click/app.* etc.) so the real
    # packages load for this test; restored on teardown so later tests see them.
    return any(key == r or key.startswith(r + ".") for r in _SHADOW_ROOTS)


@pytest.fixture
def sched_env():
    """Real in-memory DB + real app.scheduler, isolated from other tests' stubs."""
    saved = {k: sys.modules.pop(k) for k in list(sys.modules) if _is_shadow(k)}
    app_pkg = types.ModuleType("app")
    app_pkg.__path__ = [_REAL_APP_DIR]
    app_pkg.__package__ = "app"
    sys.modules["app"] = app_pkg

    import flask

    from app.extensions import db
    import app.models  # noqa: F401 — register tables

    flask_app = flask.Flask(__name__)
    flask_app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
    flask_app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    db.init_app(flask_app)
    try:
        with flask_app.app_context():
            db.create_all()
            import app.scheduler as scheduler
            import app.kalshi_trader as trader
            import app.models as models
            yield scheduler, trader, models, db
            db.session.remove()
            db.drop_all()
    finally:
        for k in [k for k in list(sys.modules) if _is_shadow(k)]:
            sys.modules.pop(k, None)
        sys.modules.update(saved)


class _Result:
    def __init__(self, signal):
        self.signal = signal


def _resting_no(db, models, ticker="KX-STALE-1"):
    t = models.LiveTrade(
        ticker=ticker, side="NO", contracts=0.0, entry_price=0.55,
        entry_price_cents=55, cost_dollars=0.0, kalshi_order_id="ord-1",
        order_status="resting", entry_at=datetime.now(timezone.utc).replace(tzinfo=None),
        resolved=False,
    )
    db.session.add(t)
    db.session.commit()
    return t


def test_cancels_resting_order_when_signal_no_longer_supports(sched_env, monkeypatch):
    scheduler, trader, models, db = sched_env
    _resting_no(db, models)

    cancelled = []
    monkeypatch.setattr(trader, "get_order_status", lambda oid: {"status": "resting", "fill_count": 0})
    monkeypatch.setattr(trader, "cancel_order", lambda oid: cancelled.append(oid))

    scheduler._cancel_stale_resting_order("KX-STALE-1", _Result("NO SIGNAL"), None)

    assert cancelled == ["ord-1"]  # cancel was requested
    row = models.LiveTrade.query.filter_by(ticker="KX-STALE-1").first()
    assert row.order_status == "unfilled"  # no longer blocks new orders / not a position
    assert row.contracts == 0


def test_keeps_resting_order_when_signal_still_supports(sched_env, monkeypatch):
    scheduler, trader, models, db = sched_env
    _resting_no(db, models)

    cancelled = []
    monkeypatch.setattr(trader, "get_order_status", lambda oid: {"status": "resting", "fill_count": 0})
    monkeypatch.setattr(trader, "cancel_order", lambda oid: cancelled.append(oid))

    scheduler._cancel_stale_resting_order("KX-STALE-1", _Result("PAPER BUY NO"), None)

    assert cancelled == []  # thesis still holds — do not cancel
    row = models.LiveTrade.query.filter_by(ticker="KX-STALE-1").first()
    assert row.order_status == "resting"


def test_does_not_cancel_if_already_filled(sched_env, monkeypatch):
    scheduler, trader, models, db = sched_env
    _resting_no(db, models)

    cancelled = []
    monkeypatch.setattr(trader, "get_order_status", lambda oid: {"status": "resting", "fill_count": 1})
    monkeypatch.setattr(trader, "cancel_order", lambda oid: cancelled.append(oid))

    scheduler._cancel_stale_resting_order("KX-STALE-1", _Result("NO SIGNAL"), None)

    assert cancelled == []  # a (partial) fill exists — leave for normal reconciliation
