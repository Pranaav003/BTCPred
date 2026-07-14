"""Deterministic backtest engine: compose signal + exit + sizing into trades."""
from __future__ import annotations

from dataclasses import dataclass

from sim.costs import CostModel, mark_price, entry_cost, exit_proceeds, fee


@dataclass
class Trade:
    ticker: str
    side: str
    contracts: int
    entry_cost_per: float
    implied_prob: float
    pnl: float
    won: bool
    exit_reason: str


def simulate(paths, signal_fn, exit_fn, sizing_fn, cfg: dict,
             cost_model: CostModel) -> list[Trade]:
    trades: list = []
    for path in paths:
        decision = signal_fn(path, cfg)
        if decision is None:
            continue
        side = decision.side
        entry_poll = path.polls[decision.entry_idx]
        e_cost = entry_cost(side, entry_poll.price_now, cost_model)
        implied = mark_price(side, entry_poll.price_now)

        win_prob = entry_poll.p_raw if side == "yes" else 1.0 - entry_poll.p_raw
        edge = abs(entry_poll.p_raw - entry_poll.price_now)
        stake = sizing_fn(edge, win_prob, e_cost, cfg)
        contracts = int(stake / e_cost) if e_cost > 0 else 0
        if contracts <= 0:
            continue

        result = exit_fn(path, decision.entry_idx, side, cfg)
        if result.exit_price is None:  # held to resolution
            won_outcome = (side == "yes" and path.final_outcome_yes == 1) or \
                          (side == "no" and path.final_outcome_yes == 0)
            proceeds_per = 1.0 if won_outcome else 0.0
        else:
            proceeds_per = exit_proceeds(side, result.exit_price, cost_model)

        gross = (proceeds_per - e_cost) * contracts
        pnl = gross - fee(gross, cost_model)
        trades.append(Trade(
            ticker=path.ticker, side=side, contracts=contracts,
            entry_cost_per=e_cost, implied_prob=implied,
            pnl=pnl, won=pnl > 0, exit_reason=result.reason,
        ))
    return trades
