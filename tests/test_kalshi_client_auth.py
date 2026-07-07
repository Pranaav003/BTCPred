"""Market-data reads must be authenticated (2026-07-07 rate-limit fix).

Bug: kalshi_client._get() fetched candlesticks/trades/markets with NO auth
headers. Kalshi throttles anonymous traffic far more aggressively than
authenticated members, so even ~2 reads per 45s poll got 429'd — starving
the scheduler of snapshots (zero trades). The authenticated /portfolio/*
calls in kalshi_trader were never rate-limited; only the anonymous reads were.

Fix: sign read requests with the same get_kalshi_headers() mechanism used for
orders, falling back to anonymous when credentials are not configured.
"""
import os
import sys
import types

import pytest

_REAL_APP_DIR = os.path.join(os.path.dirname(__file__), "..", "app")


def _is_shadow(key: str) -> bool:
    return key == "app" or key.startswith("app.")


@pytest.fixture
def kc():
    """Import the REAL app.kalshi_client without running app/__init__.py.

    Other test modules replace app.* in sys.modules with stubs at import time;
    purge those, load the real module via a stub 'app' package whose __path__
    points at the real app/ dir, then restore on teardown.
    """
    saved = {k: sys.modules.pop(k) for k in list(sys.modules) if _is_shadow(k)}
    app_pkg = types.ModuleType("app")
    app_pkg.__path__ = [_REAL_APP_DIR]
    app_pkg.__package__ = "app"
    sys.modules["app"] = app_pkg
    try:
        import app.kalshi_client as kalshi_client
        yield kalshi_client
    finally:
        for k in [k for k in list(sys.modules) if _is_shadow(k)]:
            sys.modules.pop(k, None)
        sys.modules.update(saved)


class _FakeResponse:
    status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return {"ok": True}


def test_get_attaches_signed_headers_when_configured(kc, monkeypatch):
    """When Kalshi is configured, _get must send the signed auth headers."""
    captured = {}

    def _fake_requests_get(url, params=None, headers=None, timeout=None):
        captured["headers"] = headers
        return _FakeResponse()

    fake_headers = {"KALSHI-ACCESS-KEY": "kid", "KALSHI-ACCESS-SIGNATURE": "sig"}
    sign_calls = []

    def _fake_sign(method, path):
        sign_calls.append((method, path))
        return fake_headers

    monkeypatch.setattr(kc, "is_configured", lambda: True, raising=False)
    monkeypatch.setattr(kc, "get_kalshi_headers", _fake_sign, raising=False)
    monkeypatch.setattr(kc.requests, "get", _fake_requests_get)

    kc._get("https://api.elections.kalshi.com/trade-api/v2/markets/KXBTC15M-FOO")

    assert captured["headers"] == fake_headers
    # Signed with GET over the exact request path (no host, no query).
    assert sign_calls == [("GET", "/trade-api/v2/markets/KXBTC15M-FOO")]


def test_get_is_anonymous_when_not_configured(kc, monkeypatch):
    """Without credentials, _get must fall back to anonymous (headers=None)."""
    captured = {}

    def _fake_requests_get(url, params=None, headers=None, timeout=None):
        captured["headers"] = headers
        return _FakeResponse()

    monkeypatch.setattr(kc, "is_configured", lambda: False, raising=False)
    monkeypatch.setattr(kc.requests, "get", _fake_requests_get)

    kc._get("https://api.elections.kalshi.com/trade-api/v2/markets/KXBTC15M-FOO")

    assert captured["headers"] is None
