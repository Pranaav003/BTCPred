# tests/test_hotpath_log_levels.py
# Coverage for the "performance" failure class the refutation flagged: hot poll /
# per-trade failure paths must log at WARNING with NO stack trace, so a sustained
# outage does not spam a full traceback every ~45s cycle. Also covers the
# "spec_compliance" class: benign parser swallows log at debug (still observable
# when LOG_LEVEL=DEBUG), genuine issues at warning.
import logging


def test_get_balance_error_is_warning_not_exception(monkeypatch, caplog):
    from app import kalshi_trader

    monkeypatch.setattr(kalshi_trader, "is_configured", lambda: True)
    monkeypatch.setattr(kalshi_trader, "get_kalshi_headers", lambda *a, **k: {"h": "1"})

    def _boom(*a, **k):
        raise ConnectionError("down")

    monkeypatch.setattr(kalshi_trader.requests, "get", _boom)

    with caplog.at_level(logging.DEBUG):
        assert kalshi_trader.get_balance() is None

    records = [r for r in caplog.records if r.name == "app.kalshi_trader"]
    assert records
    # Hot path: warning, and crucially NO traceback attached (exc_info is None).
    assert all(r.levelno == logging.WARNING for r in records)
    assert all(r.exc_info is None for r in records)


def test_normalize_price_swallow_logs_at_debug(monkeypatch, caplog):
    from app import kalshi_client

    with caplog.at_level(logging.DEBUG, logger="app.kalshi_client"):
        result = kalshi_client._normalize_price("not-a-number")

    assert result is None
    debug_recs = [r for r in caplog.records if r.levelno == logging.DEBUG]
    assert any("normalize price" in r.getMessage().lower() for r in debug_recs)


def test_normalize_price_swallow_invisible_at_info(caplog):
    # Documents the intentional trade-off: benign parser fallbacks are silent at
    # the default INFO level (no noise), observable only when LOG_LEVEL=DEBUG.
    from app import kalshi_client

    with caplog.at_level(logging.INFO, logger="app.kalshi_client"):
        assert kalshi_client._normalize_price("nope") is None
    assert [r for r in caplog.records if r.levelno >= logging.INFO] == []
