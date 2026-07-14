# tests/sim/test_signals.py
from sim.data import MarketPath, Poll
from sim.signals import ensemble_signal, mean_reversion_signal, EntryDecision, SIGNALS


def _cfg(**over):
    base = dict(mispricing_threshold=0.25, yes_cutoff=0.72, max_entry_yes=0.65,
                max_entry_no=0.80, no_max_p_raw=0.20, cutoff_buffer=0.03,
                min_entry_price=0.05, min_seconds=60, max_seconds=300,
                mr_return_5m=40.0, mr_price_floor=0.60)
    base.update(over)
    return base


def test_ensemble_bearish_gap_triggers_no():
    # market 29%, model 12% -> bearish gap 0.17 >= 0.25? no. Use bigger gap.
    path = MarketPath("A", 300, 1000, 0, [
        Poll(200, 0.40, 0.12, {}),  # gap = 0.12-0.40 = -0.28 -> bearish; p_raw<0.20 ok
    ])
    d = ensemble_signal(path, _cfg())
    assert isinstance(d, EntryDecision)
    assert d.side == "no" and d.entry_idx == 0


def test_ensemble_no_blocked_when_praw_too_high():
    path = MarketPath("A", 300, 1000, 0, [
        Poll(200, 0.70, 0.35, {}),  # bearish gap 0.35 but p_raw 0.35 >= no_max_p_raw
    ])
    assert ensemble_signal(path, _cfg()) is None


def test_ensemble_bullish_mispricing_triggers_yes():
    path = MarketPath("A", 300, 1000, 1, [
        Poll(200, 0.30, 0.60, {}),  # gap +0.30 >= 0.25 and p_raw>=0.5, entry 0.30 <= 0.65
    ])
    d = ensemble_signal(path, _cfg())
    assert d.side == "yes" and d.entry_idx == 0


def test_ensemble_respects_time_window_and_picks_earliest():
    path = MarketPath("A", 300, 1000, 0, [
        Poll(400, 0.40, 0.10, {}),  # outside window (>300)
        Poll(250, 0.40, 0.10, {}),  # first in-window bearish -> pick this
        Poll(120, 0.40, 0.10, {}),
    ])
    d = ensemble_signal(path, _cfg())
    assert d.entry_idx == 1 and d.side == "no"


def test_mean_reversion_fades_big_up_move():
    # large positive return_5m + expensive YES -> fade with NO
    path = MarketPath("A", 300, 1000, 0, [
        Poll(200, 0.70, 0.50, {"return_5m": 55.0}),
    ])
    d = mean_reversion_signal(path, _cfg())
    assert d.side == "no" and d.entry_idx == 0


def test_registry():
    assert SIGNALS["ensemble"] is ensemble_signal
    assert SIGNALS["mean_reversion"] is mean_reversion_signal
