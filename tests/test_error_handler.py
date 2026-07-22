# tests/test_error_handler.py
# Verifies the global @app.errorhandler(Exception): graceful JSON for /api/*,
# HTML otherwise, and HTTPException (404) passthrough. Import create_app lazily.
#
# Flask sets PROPAGATE_EXCEPTIONS=True whenever TESTING/DEBUG is true, which
# routes unhandled exceptions to the handle_exception path that RE-RAISES rather
# than invoking a registered errorhandler in some cases. We set it False PER
# TEST (not baked into TestingConfig) so the global handler is actually exercised
# and the 500 response can be asserted.
import logging

import pytest


@pytest.fixture
def app_with_boom():
    from app import create_app

    application = create_app("testing")
    application.config["PROPAGATE_EXCEPTIONS"] = False

    @application.route("/api/_boom")
    def _api_boom():
        raise RuntimeError("kaboom")

    @application.route("/_boom_html")
    def _html_boom():
        raise RuntimeError("kaboom")

    return application


@pytest.fixture
def client(app_with_boom):
    return app_with_boom.test_client()


def test_api_route_returns_500_json(client):
    resp = client.get("/api/_boom")
    assert resp.status_code == 500
    assert resp.is_json
    body = resp.get_json()
    assert body["error"] == "internal error"
    assert body["status"] == 500


def test_non_api_route_returns_500_html_not_json(client):
    resp = client.get("/_boom_html")
    assert resp.status_code == 500
    # Not JSON — a minimal HTML/text message.
    assert not resp.is_json
    assert b"error" in resp.data.lower() or b"wrong" in resp.data.lower()


def test_unhandled_exception_is_logged(client, caplog):
    with caplog.at_level(logging.ERROR):
        client.get("/api/_boom")
    # logger.exception emits at ERROR with traceback text.
    assert any(rec.levelno >= logging.ERROR for rec in caplog.records)
    assert any("kaboom" in (rec.exc_text or "") or rec.exc_info for rec in caplog.records)


def test_404_still_returns_404_not_500(client):
    resp = client.get("/api/does-not-exist-xyz")
    assert resp.status_code == 404  # HTTPException must pass through unchanged


def test_405_method_not_allowed_passes_through(client):
    # /api/_boom is GET-only; POSTing yields a 405 HTTPException, not a 500.
    resp = client.post("/api/_boom")
    assert resp.status_code == 405


def test_configure_logging_called_by_create_app():
    import logging as _logging

    from app import create_app

    create_app("testing")
    root = _logging.getLogger()
    assert any(getattr(h, "name", None) == "btcpred-stdout" for h in root.handlers)
