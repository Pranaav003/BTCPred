"""Position sizing functions. All return a dollar stake (0.0 = skip the trade).

`entry_cost` is the per-contract cost (0-1). Payoff if win = (1 - entry_cost)
per contract; loss if lose = entry_cost per contract.
"""
from __future__ import annotations

from sim.metrics import breakeven_win_rate


def flat_size(edge: float, win_prob: float, entry_cost: float, cfg: dict) -> float:
    return cfg["base_size"]


def fractional_kelly_size(edge: float, win_prob: float, entry_cost: float,
                          cfg: dict) -> float:
    if entry_cost <= 0 or entry_cost >= 1:
        return 0.0
    b = (1.0 - entry_cost) / entry_cost  # net odds per unit staked
    p = win_prob
    kelly = (b * p - (1.0 - p)) / b
    if kelly <= 0:
        return 0.0
    # base_size scales the Kelly fraction (it is a size unit, not bankroll); max_size caps the result.
    stake = cfg["base_size"] * cfg["kelly_fraction"] * kelly * 10.0
    return min(stake, cfg["max_size"])


def payoff_aware_size(edge: float, win_prob: float, entry_cost: float,
                      cfg: dict) -> float:
    avg_win = 1.0 - entry_cost
    avg_loss = entry_cost
    be = breakeven_win_rate(avg_win, avg_loss)
    if win_prob <= be:
        return 0.0
    margin = win_prob - be
    return min(cfg["base_size"] * (1.0 + margin * 5.0), cfg["max_size"])


SIZERS = {
    "flat": flat_size,
    "kelly": fractional_kelly_size,
    "payoff_aware": payoff_aware_size,
}
