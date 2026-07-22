# tests/test_targeted_logging_fixes.py
# Regression tests for the Critical targeted fixes:
#  * kalshi_trader.get_open_positions logs (and still returns []) on non-200 and
#    on exception — but at WARNING level (hot per-trade path, no stack-trace spam).
#  * db_helpers.get_setting logs warning on first retry, error on the second.
#  * db_helpers.export_training_data logs ONE post-loop summary of skipped rows,
#    not one line per corrupt row.
import logging

import pytest


# --- kalshi_trader.get_open_positions ---------------------------------------

def test_get_open_positions_logs_and_returns_empty_on_non_200(monkeypatch, caplog):
    from app import kalshi_trader

    class _Resp:
        status_code = 503
        text = "unavailable"

        def json(self):
            return {}

    monkeypatch.setattr(kalshi_trader, "is_configured", lambda: True)
    monkeypatch.setattr(kalshi_trader, "get_kalshi_headers", lambda *a, **k: {"h": "1"})
    monkeypatch.setattr(kalshi_trader.requests, "get", lambda *a, **k: _Resp())

    with caplog.at_level(logging.WARNING):
        result = kalshi_trader.get_open_positions()

    assert result == []
    assert any(rec.levelno == logging.WARNING for rec in caplog.records)


def test_get_open_positions_logs_and_returns_empty_on_exception(monkeypatch, caplog):
    from app import kalshi_trader

    def _boom(*a, **k):
        raise ConnectionError("network down")

    monkeypatch.setattr(kalshi_trader, "is_configured", lambda: True)
    monkeypatch.setattr(kalshi_trader, "get_kalshi_headers", lambda *a, **k: {"h": "1"})
    monkeypatch.setattr(kalshi_trader.requests, "get", _boom)

    with caplog.at_level(logging.WARNING):
        result = kalshi_trader.get_open_positions()

    assert result == []
    assert any(rec.levelno == logging.WARNING for rec in caplog.records)
    # Hot path: must NOT emit at ERROR (no stack-trace spam every poll).
    assert all(rec.levelno < logging.ERROR for rec in caplog.records)


# --- db_helpers.get_setting --------------------------------------------------

def test_get_setting_logs_warning_then_error_across_retries(app, monkeypatch, caplog):
    from app import db_helpers
    from sqlalchemy.exc import SQLAlchemyError

    calls = {"n": 0}

    class _Query:
        def filter_by(self, **kw):
            return self

        def first(self):
            calls["n"] += 1
            raise SQLAlchemyError(f"db error {calls['n']}")

    monkeypatch.setattr(db_helpers.AppSettings, "query", _Query())

    with caplog.at_level(logging.WARNING):
        value = db_helpers.get_setting("some_key", default="fallback")

    assert value == "fallback"  # returns default after both retries fail
    levels = sorted(rec.levelno for rec in caplog.records)
    assert logging.WARNING in levels  # first retry
    assert logging.ERROR in levels  # second failure


# --- db_helpers.export_training_data (post-loop summary) --------------------

def test_export_training_data_logs_single_summary_not_per_row(app, monkeypatch, caplog, tmp_path):
    from app import db_helpers

    class _Sig:
        raw_features_json = "{not valid json"  # forces the except: skipped += 1 path
        market = None
        p_market = 0.5
        p_raw = 0.5
        logged_at = None
        agreement_region = "x"
        signal = "NO SIGNAL"

    class _Query:
        def join(self, *a, **k):
            return self

        def filter(self, *a, **k):
            return self

        def options(self, *a, **k):
            return self

        def order_by(self, *a, **k):
            return self

        def yield_per(self, n):
            return [_Sig(), _Sig(), _Sig()]  # 3 corrupt rows

    monkeypatch.setattr(db_helpers.Signal, "query", _Query())

    out = tmp_path / "out.csv"
    with caplog.at_level(logging.WARNING):
        rows, skipped = db_helpers.export_training_data(str(out))

    assert skipped == 3
    # Exactly ONE summary log line mentioning the skip count — not one per row.
    summary = [r for r in caplog.records if "skip" in r.getMessage().lower()]
    assert len(summary) == 1
    assert "3" in summary[0].getMessage()
