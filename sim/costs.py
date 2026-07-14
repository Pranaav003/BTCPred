# sim/costs.py
"""Single source of truth for entry/exit costs and fees on Kalshi contracts."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CostModel:
    spread_offset_yes: float = 0.02
    spread_offset_no_low: float = 0.05
    spread_offset_no_high: float = 0.03
    no_low_threshold: float = 0.40
    exit_spread: float = 0.02
    fee_rate: float = 0.01
    max_price: float = 0.99


def mark_price(side: str, price_now: float) -> float:
    """Current contract mark for the given side (0-1)."""
    if side not in ("yes", "no"):
        raise ValueError(f"side must be 'yes' or 'no', got {side!r}")
    return price_now if side == "yes" else 1.0 - price_now


def entry_cost(side: str, price_now: float, cfg: CostModel) -> float:
    """Per-contract cost to enter, incl. aggressive-fill offset, capped."""
    mark = mark_price(side, price_now)
    if side == "yes":
        offset = cfg.spread_offset_yes
    else:
        offset = (cfg.spread_offset_no_low if mark <= cfg.no_low_threshold
                  else cfg.spread_offset_no_high)
    return min(cfg.max_price, mark + offset)


def exit_proceeds(side: str, price_now: float, cfg: CostModel) -> float:
    """Per-contract proceeds from selling at the current mark, net of spread."""
    return max(0.0, mark_price(side, price_now) - cfg.exit_spread)


def fee(gross_gain: float, cfg: CostModel) -> float:
    """Fee applied only to positive gross gains."""
    return gross_gain * cfg.fee_rate if gross_gain > 0 else 0.0
