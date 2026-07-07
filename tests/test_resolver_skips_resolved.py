"""Regression test for the resolver rate-limit spiral (2026-07-07).

Bug: resolve_live_trades() re-queried Kalshi settlement/resolution for EVERY
live trade in the last 48h on every 60s cycle — including trades that were
already resolved — because its query lacked a `resolved == False` filter.
With ~99 trades this generated ~50-100 wasted API calls per minute that
competed with the scheduler's snapshot fetches, causing rate-limit 429s and
"No live snapshot available" (zero trades).

This test drives a real in-memory SQLite DB and asserts that an
already-resolved trade's ticker is never sent to the Kalshi API.

NOTE on isolation: other test modules (e.g. test_paper_trading.py) replace
`app`, `app.extensions`, `app.models`, `flask*` in sys.modules with lightweight
stubs at import time and never restore them. This test needs the REAL modules,
so the fixture purges any polluted entries, loads the real submodules via a
stub `app` package whose __path__ points at the real app/ dir (avoiding the
heavy app/__init__.py which imports sklearn/xgboost), then restores sys.modules
on teardown so subsequent tests see the state they expect.
"""
import importlib
import os
import sys
import types
from datetime import datetime, timedelta, timezone

import pytest

_REAL_APP_DIR = os.path.join(os.path.dirname(__file__), "..", "app")


_SHADOW_ROOTS = (
    "app",
    "flask",
    "flask_sqlalchemy",
    "flask_migrate",
    "flask_wtf",
    "sqlalchemy",
    "click",
    "dotenv",
)


def _is_shadow(key: str) -> bool:
    """True for modules other test files replace with fakes that we need real.

    Other suites stub these (and their transitive deps) at import time to run
    without the real packages installed; we need the real ones, so we evict and
    reload them for the duration of this test, then restore on teardown.
    """
    return any(key == root or key.startswith(root + ".") for root in _SHADOW_ROOTS)


@pytest.fixture
def app_ctx():
    """Minimal Flask app bound to an in-memory SQLite DB with the REAL models."""
    # Snapshot and evict any stub modules a prior test file left behind (fake
    # flask/sqlalchemy/app.*), so the REAL packages load fresh for this test.
    saved = {k: sys.modules.pop(k) for k in list(sys.modules) if _is_shadow(k)}

    # Install a stub 'app' package whose __path__ points at the real app dir so
    # `import app.extensions` loads the real file without running app/__init__.py.
    app_pkg = types.ModuleType("app")
    app_pkg.__path__ = [_REAL_APP_DIR]
    app_pkg.__package__ = "app"
    sys.modules["app"] = app_pkg

    import flask
    importlib.import_module("flask_sqlalchemy")  # ensure the REAL extension is loaded

    from app.extensions import db
    import app.models  # noqa: F401 — registers tables on db.metadata before create_all

    flask_app = flask.Flask(__name__)
    flask_app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
    flask_app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    db.init_app(flask_app)
    try:
        with flask_app.app_context():
            db.create_all()
            yield db
            db.session.remove()
            db.drop_all()
    finally:
        # Remove the real modules we loaded and restore whatever the rest of
        # the suite left in place (its fakes), so later stub-based tests pass.
        for k in [k for k in list(sys.modules) if _is_shadow(k)]:
            sys.modules.pop(k, None)
        sys.modules.update(saved)


def _make_trade(db, models, ticker, resolved):
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    trade = models.LiveTrade(
        ticker=ticker,
        side="YES",
        contracts=1.0,
        entry_price=0.50,
        entry_price_cents=50,
        cost_dollars=0.50,
        kalshi_order_id=f"order-{ticker}",
        order_status="placed",
        entry_at=now - timedelta(hours=1),  # within the 48h window
        resolved=resolved,
    )
    db.session.add(trade)
    db.session.commit()
    return trade


def test_resolve_live_trades_skips_already_resolved_trades(app_ctx, monkeypatch):
    """Already-resolved trades must NOT trigger any Kalshi settlement/resolution call."""
    db = app_ctx
    import app.models as models
    import app.resolver as resolver
    import app.kalshi_trader as trader

    _make_trade(db, models, "KX-RESOLVED-OLD", resolved=True)
    _make_trade(db, models, "KX-UNRESOLVED-NEW", resolved=False)

    queried_tickers = []

    def _fake_settlement(ticker):
        queried_tickers.append(ticker)
        return None  # force fallthrough to get_market_resolution

    def _fake_resolution(ticker):
        queried_tickers.append(ticker)
        return {"resolved": False}  # not settled → no DB change

    monkeypatch.setattr(trader, "is_configured", lambda: True)
    monkeypatch.setattr(trader, "cancel_order", lambda *a, **k: None)
    monkeypatch.setattr(trader, "get_settlement_for_ticker", _fake_settlement)
    monkeypatch.setattr(resolver, "get_market_resolution", _fake_resolution)

    resolver.resolve_live_trades()

    # Sanity: the unresolved trade SHOULD be queried (proves the 48h window works).
    assert "KX-UNRESOLVED-NEW" in queried_tickers
    # The bug: the already-resolved trade must NOT hit the Kalshi API at all.
    assert "KX-RESOLVED-OLD" not in queried_tickers
