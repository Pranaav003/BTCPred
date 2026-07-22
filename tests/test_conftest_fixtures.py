import os

import pytest

# /api/health reports 503 "degraded" when no model is loaded (correct liveness
# behavior). The model .pkl is gitignored, so a fresh CI checkout has none —
# skip this healthy-path assertion there, matching tests/test_e2e_pipeline.py.
MODEL_PRESENT = os.path.exists("raw_feature_model.pkl")


@pytest.mark.skipif(not MODEL_PRESENT, reason="raw_feature_model.pkl artifact required")
def test_client_fixture_hits_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.get_json()["status"] in ("ok", "healthy", "up")


def test_app_fixture_has_app_context(app):
    from app.db_helpers import get_setting
    # seeded default exists (seed_default_settings ran in create_app)
    assert get_setting("signal_mode") is not None
