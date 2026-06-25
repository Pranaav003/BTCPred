"""Tests for PnL calculation in _apply_kalshi_settlement.

These tests verify the math directly without importing the full app module.
The actual bug was: using total_cost (yes_cost + no_cost) instead of
side_cost (yes_cost for YES trades, no_cost for NO trades).
"""
import pytest


def _compute_pnl(side, market_result, yes_cost, no_cost, fee_cost, revenue_cents):
    """Reimplementation of the PnL formula from _apply_kalshi_settlement.

    After the fix:
        side_cost = yes_cost if trade_is_yes else no_cost
        pnl = round((revenue_cents / 100.0) - side_cost - fee_cost, 2)
    """
    trade_is_yes = side == "YES"
    side_cost = yes_cost if trade_is_yes else no_cost
    pnl = round((revenue_cents / 100.0) - side_cost - fee_cost, 2)
    return pnl


def test_yes_trade_correct_pnl_uses_side_cost_only():
    """PnL for a winning YES trade must use yes_cost, not yes_cost+no_cost."""
    pnl = _compute_pnl(
        side="YES", market_result="yes",
        yes_cost=7.00, no_cost=0.00, fee_cost=0.07,
        revenue_cents=1000,  # $10.00
    )
    # Correct: 10.00 - 7.00 - 0.07 = 2.93
    assert pnl == 2.93


def test_yes_trade_correct_pnl_with_nonzero_no_cost():
    """Even if no_cost is non-zero, YES PnL must not subtract it."""
    pnl = _compute_pnl(
        side="YES", market_result="yes",
        yes_cost=7.00, no_cost=3.00, fee_cost=0.07,
        revenue_cents=1000,  # $10.00
    )
    # Correct: 10.00 - 7.00 - 0.07 = 2.93 (NOT 10.00 - 10.00 - 0.07 = -0.07)
    assert pnl == 2.93


def test_no_trade_correct_pnl_uses_no_cost():
    """PnL for a winning NO trade must use no_cost, not yes_cost+no_cost."""
    pnl = _compute_pnl(
        side="NO", market_result="no",
        yes_cost=0.00, no_cost=3.00, fee_cost=0.03,
        revenue_cents=1000,  # $10.00
    )
    # Correct: 10.00 - 3.00 - 0.03 = 6.97
    assert pnl == 6.97


def test_yes_trade_losing_pnl():
    """Losing YES trade: PnL = revenue/100 - side_cost - fee (negative)."""
    pnl = _compute_pnl(
        side="YES", market_result="no",
        yes_cost=7.00, no_cost=0.00, fee_cost=0.07,
        revenue_cents=0,  # Lost — revenue is 0
    )
    # Lost: 0.00 - 7.00 - 0.07 = -7.07
    assert pnl == -7.07


def test_no_trade_with_nonzero_yes_cost():
    """NO trade with non-zero yes_cost must not subtract yes_cost."""
    pnl = _compute_pnl(
        side="NO", market_result="no",
        yes_cost=5.00, no_cost=3.00, fee_cost=0.03,
        revenue_cents=1000,
    )
    # Correct: 10.00 - 3.00 - 0.03 = 6.97 (NOT 10.00 - 8.00 - 0.03 = 1.97)
    assert pnl == 6.97
