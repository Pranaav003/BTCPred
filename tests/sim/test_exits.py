# tests/sim/test_exits.py
from sim.data import MarketPath, Poll
from sim.exits import (hold_to_resolution, take_profit_stop_loss, trailing_stop,
                       ExitResult, EXITS)


def _path(prices):
    polls = [Poll(300 - i * 30, p, 0.5, {}) for i, p in enumerate(prices)]
    return MarketPath("A", 300, 1000, 1, polls)


def test_hold_to_resolution():
    r = hold_to_resolution(_path([0.4, 0.5, 0.6]), 0, "yes", {})
    assert r.exit_price is None and r.reason == "resolution"
    assert r.exit_idx == 2  # last poll index


def test_stop_loss_triggers_for_yes():
    # YES entered at mark 0.50; price falls; SL 0.10 -> exit when mark <= 0.40
    r = take_profit_stop_loss(_path([0.50, 0.45, 0.38, 0.30]), 0, "yes",
                              {"tp_abs": 0.0, "sl_abs": 0.10})
    assert r.reason == "stop_loss" and r.exit_idx == 2
    assert r.exit_price == 0.38


def test_take_profit_triggers_for_no():
    # NO entered at price_now 0.50 -> mark 0.50; NO mark rises as price_now FALLS.
    # price_now 0.50 -> 0.30 => NO mark 0.50 -> 0.70, gain +0.20 >= tp 0.15
    r = take_profit_stop_loss(_path([0.50, 0.40, 0.30]), 0, "no",
                              {"tp_abs": 0.15, "sl_abs": 0.0})
    assert r.reason == "take_profit" and r.exit_idx == 2
    assert r.exit_price == 0.30


def test_no_trigger_holds():
    r = take_profit_stop_loss(_path([0.50, 0.52, 0.51]), 0, "yes",
                              {"tp_abs": 0.20, "sl_abs": 0.20})
    assert r.reason == "resolution" and r.exit_price is None


def test_trailing_stop_locks_gain():
    # YES entry 0.50, rises to 0.70 (peak), trail 0.10 -> exit when mark <= 0.60
    r = trailing_stop(_path([0.50, 0.70, 0.58]), 0, "yes", {"trail_abs": 0.10})
    assert r.reason == "trailing" and r.exit_idx == 2 and r.exit_price == 0.58


def test_registry():
    assert EXITS["hold"] is hold_to_resolution
    assert EXITS["tp_sl"] is take_profit_stop_loss
    assert EXITS["trailing"] is trailing_stop
