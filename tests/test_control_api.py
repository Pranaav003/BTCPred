# tests/test_control_api.py
def test_control_state_shape(client):
    resp = client.get("/api/control/state", headers={"Accept": "application/json"})
    assert resp.status_code == 200
    data = resp.get_json()
    for key in ("mode", "scheduler_running", "paper_trading_enabled",
                "auto_trade_enabled", "signal_mode", "mispricing_threshold",
                "breakeven_win_rate", "trades_today", "paper_pnl_today"):
        assert key in data
    assert data["mode"] == "paper"  # default seed = live off
    assert data["paper_trading_enabled"] is True


def test_apply_defaults_sets_validated_paper_config(client):
    resp = client.post("/api/control/apply-defaults",
                       headers={"Accept": "application/json"})
    assert resp.status_code == 200
    assert "updated" in resp.get_json()
    from app.db_helpers import get_setting
    assert get_setting("signal_mode") == "ensemble"
    assert float(get_setting("mispricing_threshold")) == 0.25
    assert get_setting("paper_trading_enabled") == "true"
    assert get_setting("auto_trade_enabled") == "true"
    assert get_setting("live_trading_enabled") == "false"
