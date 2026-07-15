# tests/test_api_settings.py
def test_get_settings_returns_dict(client):
    resp = client.get("/api/settings", headers={"Accept": "application/json"})
    assert resp.status_code == 200
    assert isinstance(resp.get_json(), dict)


def test_post_settings_roundtrip(client):
    resp = client.post(
        "/api/settings",
        json={"mispricing_threshold": 0.25},
        headers={"Accept": "application/json"},
    )
    assert resp.status_code == 200
    from app.db_helpers import get_setting
    assert float(get_setting("mispricing_threshold")) == 0.25
