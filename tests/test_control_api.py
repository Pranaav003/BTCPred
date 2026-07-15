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
