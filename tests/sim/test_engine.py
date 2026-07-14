# tests/sim/test_engine.py
import pytest
from sim.data import MarketPath, Poll
from sim.costs import CostModel
from sim.engine import simulate, Trade
from sim.signals import ensemble_signal
from sim.exits import hold_to_resolution, take_profit_stop_loss
from sim.sizing import flat_size


def _cfg(**o):
    b = dict(mispricing_threshold=0.25, yes_cutoff=0.72, max_entry_yes=0.65,
             max_entry_no=0.80, no_max_p_raw=0.20, cutoff_buffer=0.03,
             min_entry_price=0.05, min_seconds=60, max_seconds=300,
             mr_return_5m=40.0, mr_price_floor=0.60,
             base_size=1.0, kelly_fraction=0.5, max_size=20.0,
             tp_abs=0.0, sl_abs=0.0, trail_abs=0.0)
    b.update(o)
    return b


def test_no_trade_when_signal_none():
    path = MarketPath("A", 300, 1000, 1, [Poll(200, 0.50, 0.50, {})])  # no signal
    trades = simulate([path], ensemble_signal, hold_to_resolution, flat_size,
                      _cfg(), CostModel())
    assert trades == []


def test_winning_no_held_to_resolution():
    # bearish gap NO, market resolves NO (final_outcome_yes=0) -> win
    path = MarketPath("A", 300, 1000, 0, [Poll(200, 0.40, 0.10, {})])
    trades = simulate([path], ensemble_signal, hold_to_resolution, flat_size,
                      _cfg(), CostModel())
    assert len(trades) == 1
    t = trades[0]
    assert isinstance(t, Trade) and t.side == "no" and t.won is True
    # entry cost per: NO mark = 0.60 (>0.40) -> +0.03 = 0.63; contracts=int(1/0.63)=1
    assert t.contracts == 1
    assert t.entry_cost_per == pytest.approx(0.63)
    # proceeds 1.0, gross = 1.0-0.63 = 0.37, fee 1% of 0.37 -> pnl 0.3663
    assert t.pnl == pytest.approx(0.37 * 0.99, abs=1e-6)
    assert t.implied_prob == pytest.approx(0.60)


def test_losing_trade_full_loss():
    # bearish NO but market resolves YES -> loss of full entry cost
    path = MarketPath("A", 300, 1000, 1, [Poll(200, 0.40, 0.10, {})])
    trades = simulate([path], ensemble_signal, hold_to_resolution, flat_size,
                      _cfg(), CostModel())
    t = trades[0]
    assert t.won is False
    assert t.pnl == pytest.approx(-0.63)  # 1 contract * -entry_cost


def test_zero_size_skips_trade():
    # sizer returns 0.0 stake -> contracts 0 -> no trade recorded
    path = MarketPath("A", 300, 1000, 0, [Poll(200, 0.40, 0.10, {})])
    trades = simulate([path], ensemble_signal, hold_to_resolution,
                      lambda edge, wp, ec, cfg: 0.0, _cfg(), CostModel())
    assert trades == []


def test_stop_loss_caps_loss_smaller_than_full():
    # NO entered mark 0.60; price_now rises 0.40->0.55 => NO mark 0.60->0.45,
    # gain -0.15 <= -sl(0.10) -> exit early; resolves YES so hold would be full loss.
    path = MarketPath("A", 300, 1000, 1, [
        Poll(200, 0.40, 0.10, {}),
        Poll(150, 0.55, 0.10, {}),
    ])
    trades = simulate([path], ensemble_signal, take_profit_stop_loss, flat_size,
                      _cfg(sl_abs=0.10), CostModel())
    t = trades[0]
    assert t.exit_reason == "stop_loss"
    # exit proceeds: NO mark at 0.55 = 0.45, minus exit spread 0.02 = 0.43
    # gross = 0.43 - 0.63 = -0.20 -> pnl -0.20 (smaller than full -0.63)
    assert t.pnl == pytest.approx(-0.20)
    assert t.pnl > -0.63
