from sim.promote import config_to_settings, promotion_candidate


def test_config_to_settings_maps_known_keys():
    cfg = {"signal": "ensemble", "mispricing_threshold": 0.2,
           "max_entry_yes": 0.6, "max_entry_no": 0.75, "no_max_p_raw": 0.18,
           "yes_cutoff": 0.7}
    s = config_to_settings(cfg)
    assert s["mispricing_threshold"] == "0.2"
    assert s["max_entry_price_yes"] == "0.6"
    assert s["max_entry_price_no"] == "0.75"
    assert s["no_max_p_raw"] == "0.18"
    assert s["yes_cutoff"] == "0.7"
    assert "signal" not in s  # not a live setting key


def test_config_to_settings_only_present_keys():
    s = config_to_settings({"mispricing_threshold": 0.25})
    assert s == {"mispricing_threshold": "0.25"}


def test_promotion_candidate_prefers_passing():
    board = [
        {"passed": False, "score": 5.0, "config": {"a": 1}},
        {"passed": True, "score": 1.0, "config": {"a": 2}},
    ]
    assert promotion_candidate(board)["config"] == {"a": 2}


def test_promotion_candidate_none_when_no_pass():
    assert promotion_candidate([{"passed": False, "score": 1.0}]) is None


def test_promotion_candidate_empty_board():
    assert promotion_candidate([]) is None
