# sim/signals.py
"""Pluggable entry-signal functions. Each returns the earliest qualifying entry."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EntryDecision:
    entry_idx: int
    side: str  # "yes" or "no"


def _in_window(seconds: int, cfg: dict) -> bool:
    return cfg["min_seconds"] <= seconds <= cfg["max_seconds"]


def ensemble_signal(path, cfg: dict):
    """Vectorized port of evaluate_ensemble_signal, scanning earliest-first."""
    thresh = cfg["mispricing_threshold"]
    yes_cut = cfg["yes_cutoff"]
    for idx, poll in enumerate(path.polls):
        if not _in_window(poll.seconds_to_close, cfg):
            continue
        p_market, p_raw = poll.price_now, poll.p_raw
        if abs(p_raw - yes_cut) < cfg["cutoff_buffer"]:
            continue  # noisy edge zone
        gap = p_raw - p_market
        agreement_yes = p_market >= yes_cut and p_raw >= yes_cut
        mispricing_bull = gap >= thresh and p_raw >= 0.50
        mispricing_bear = (-gap) >= thresh and p_raw < 0.50
        yes_ok = cfg["min_entry_price"] <= p_market <= cfg["max_entry_yes"]
        no_ok = cfg["min_entry_price"] <= (1.0 - p_market) <= cfg["max_entry_no"]
        no_praw_ok = p_raw < cfg["no_max_p_raw"]
        if (agreement_yes or mispricing_bull) and yes_ok:
            return EntryDecision(idx, "yes")
        if mispricing_bear and no_ok and no_praw_ok:
            return EntryDecision(idx, "no")
    return None


def mean_reversion_signal(path, cfg: dict):
    """Fade a large recent BTC move: big up-move + pricey YES -> buy NO, and vice-versa."""
    for idx, poll in enumerate(path.polls):
        if not _in_window(poll.seconds_to_close, cfg):
            continue
        r5 = poll.features.get("return_5m", 0.0)
        p_market = poll.price_now
        if r5 >= cfg["mr_return_5m"] and p_market >= cfg["mr_price_floor"]:
            return EntryDecision(idx, "no")
        if r5 <= -cfg["mr_return_5m"] and p_market <= (1.0 - cfg["mr_price_floor"]):
            return EntryDecision(idx, "yes")
    return None


SIGNALS = {
    "ensemble": ensemble_signal,
    "mean_reversion": mean_reversion_signal,
}
