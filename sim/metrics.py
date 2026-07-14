# sim/metrics.py
"""P&L, risk, and calibration metrics computed from a trade ledger."""
from __future__ import annotations

import math


def breakeven_win_rate(avg_win: float, avg_loss: float) -> float:
    """Win rate needed to break even. avg_loss is a positive magnitude."""
    denom = avg_win + avg_loss
    return avg_loss / denom if denom > 0 else 0.0


def _std(xs: list) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mean = sum(xs) / n
    return math.sqrt(sum((x - mean) ** 2 for x in xs) / (n - 1))


def compute_metrics(pnls: list, contracts: list) -> dict:
    n = len(pnls)
    if n == 0:
        return {
            "n_trades": 0, "win_rate": 0.0, "total_pnl": 0.0, "avg_win": 0.0,
            "avg_loss": 0.0, "profit_factor": 0.0, "sharpe": 0.0,
            "sortino": 0.0, "max_drawdown": 0.0, "ev_per_contract": 0.0,
        }
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gross_win = sum(wins)
    gross_loss = -sum(losses)
    total = sum(pnls)

    # equity curve max drawdown
    equity, peak, max_dd = 0.0, 0.0, 0.0
    for p in pnls:
        equity += p
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)

    downside = [p for p in pnls if p < 0]
    total_contracts = sum(contracts) if contracts else 0
    return {
        "n_trades": n,
        "win_rate": len(wins) / n,
        "total_pnl": total,
        "avg_win": (gross_win / len(wins)) if wins else 0.0,
        "avg_loss": (sum(losses) / len(losses)) if losses else 0.0,
        "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else 0.0,
        "sharpe": (total / n) / _std(pnls) if _std(pnls) > 0 else 0.0,
        "sortino": (total / n) / _std(downside) if _std(downside) > 0 else 0.0,
        "max_drawdown": max_dd,
        "ev_per_contract": (total / total_contracts) if total_contracts else 0.0,
    }


def calibration(probs: list, outcomes: list) -> dict:
    n = len(probs)
    if n == 0:
        return {"brier": 0.0, "log_loss": 0.0}
    brier = sum((p - y) ** 2 for p, y in zip(probs, outcomes)) / n
    eps = 1e-12
    ll = -sum(
        y * math.log(min(max(p, eps), 1 - eps))
        + (1 - y) * math.log(min(max(1 - p, eps), 1 - eps))
        for p, y in zip(probs, outcomes)
    ) / n
    return {"brier": brier, "log_loss": ll}
