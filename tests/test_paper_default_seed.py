def test_paper_defaults_on_live_off(app):
    from app.db_helpers import get_setting
    assert get_setting("paper_trading_enabled") == "true"
    assert get_setting("auto_trade_enabled") == "true"
    assert get_setting("live_trading_enabled") == "false"
    assert get_setting("mispricing_threshold") == "0.25"
    assert get_setting("scheduler_running") == "true"
