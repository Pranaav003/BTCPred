"""Pluggable exit policies operating on a market's intra-market price path.

All comparisons are in *position mark* terms (NO mark = 1 - price_now), so a
gain is favorable regardless of side. Exits may only read polls at or after the
entry index (no look-ahead).
"""
from __future__ import annotations

from dataclasses import dataclass

from sim.costs import mark_price


@dataclass
class ExitResult:
    exit_idx: int
    exit_price: float | None  # None => held to resolution (use final outcome)
    reason: str


def hold_to_resolution(path, entry_idx: int, side: str, cfg: dict) -> ExitResult:
    return ExitResult(exit_idx=len(path.polls) - 1, exit_price=None,
                      reason="resolution")


def take_profit_stop_loss(path, entry_idx: int, side: str, cfg: dict) -> ExitResult:
    tp = cfg.get("tp_abs", 0.0)
    sl = cfg.get("sl_abs", 0.0)
    entry_mark = mark_price(side, path.polls[entry_idx].price_now)
    for idx in range(entry_idx + 1, len(path.polls)):
        poll = path.polls[idx]
        gain = mark_price(side, poll.price_now) - entry_mark
        if tp > 0 and gain >= tp:
            return ExitResult(idx, poll.price_now, "take_profit")
        if sl > 0 and gain <= -sl:
            return ExitResult(idx, poll.price_now, "stop_loss")
    return ExitResult(len(path.polls) - 1, None, "resolution")


def trailing_stop(path, entry_idx: int, side: str, cfg: dict) -> ExitResult:
    trail = cfg.get("trail_abs", 0.0)
    peak = mark_price(side, path.polls[entry_idx].price_now)
    for idx in range(entry_idx + 1, len(path.polls)):
        poll = path.polls[idx]
        mark = mark_price(side, poll.price_now)
        peak = max(peak, mark)
        if trail > 0 and mark <= peak - trail:
            return ExitResult(idx, poll.price_now, "trailing")
    return ExitResult(len(path.polls) - 1, None, "resolution")


EXITS = {
    "hold": hold_to_resolution,
    "tp_sl": take_profit_stop_loss,
    "trailing": trailing_stop,
}
