def test_client_fixture_hits_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.get_json()["status"] in ("ok", "healthy", "up")


def test_app_fixture_has_app_context(app):
    from app.db_helpers import get_setting
    # seeded default exists (seed_default_settings ran in create_app)
    assert get_setting("signal_mode") is not None
