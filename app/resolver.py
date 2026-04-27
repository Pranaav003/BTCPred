"""Market resolution and signal PnL backfill utilities."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from app.extensions import db
from app.db_helpers import resolve_paper_trades
from app.kalshi_client import get_market_resolution
from app.models import Market, Signal

logger = logging.getLogger(__name__)


def _utc_iso_z(value: datetime | None) -> str | None:
    """Serialize datetime as UTC ISO string with trailing Z."""
    if value is None:
        return None
    if value.tzinfo is None:
        return f"{value.isoformat()}Z"
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def compute_pnl(signal_str, outcome_yes, p_market, contracts=1.0):
    """Compute paper PnL and correctness for a resolved signal."""
    signal_name = (signal_str or "NO SIGNAL").strip().upper()
    market_prob = float(p_market) if p_market is not None else 0.0
    contracts_value = float(contracts or 0.0)
    if contracts_value <= 0:
        contracts_value = 1.0

    if signal_name == "PAPER BUY YES":
        cost_per = market_prob
        payout_per = 1.0 if outcome_yes else 0.0
        pnl = (payout_per - cost_per) * contracts_value
        outcome_correct = bool(outcome_yes is True)
    elif signal_name == "PAPER BUY NO":
        # NO contract price is inverse of YES probability.
        cost_per = 1.0 - market_prob
        payout_per = 1.0 if not outcome_yes else 0.0
        pnl = (payout_per - cost_per) * contracts_value
        outcome_correct = bool(outcome_yes is False)
    else:
        pnl = 0.0
        outcome_correct = None

    return float(pnl), outcome_correct


def resolve_market(market):
    """Resolve a single unresolved market and back-fill signal outcomes."""
    result = get_market_resolution(market.ticker)
    if result is None:
        logger.warning("Resolution fetch failed for market %s", market.ticker)
        return False

    if result.get("resolved") is not True:
        return False

    result_value = result.get("result")
    if isinstance(result_value, str):
        lowered = result_value.strip().lower()
        if lowered == "yes":
            outcome_yes = True
        elif lowered == "no":
            outcome_yes = False
        else:
            outcome_yes = None
    else:
        outcome_yes = None

    if outcome_yes is None:
        final_price = result.get("final_price")
        if final_price is None:
            return False
        outcome_yes = float(final_price) >= 0.5

    market.resolved = True
    market.final_outcome_yes = bool(outcome_yes)
    market.resolution_price = result.get("final_price")

    signals = Signal.query.filter_by(market_id=market.id).all()
    for signal in signals:
        pnl, correct = compute_pnl(signal.signal, outcome_yes, signal.p_market, contracts=1.0)
        signal.resolved = True
        signal.pnl = pnl
        signal.outcome_correct = correct

    db.session.commit()
    resolved_trades = resolve_paper_trades(market)

    logger.info(
        "Resolved %s: outcome=%s, %s signals updated, %s paper trades resolved",
        market.ticker,
        "YES" if outcome_yes else "NO",
        len(signals),
        resolved_trades,
    )
    return True


def resolve_pending_markets():
    """Resolve all past-close unresolved markets. Never raises."""
    try:
        now_utc = datetime.now(timezone.utc)
        pending = Market.query.filter(
            Market.resolved.is_(False),
            Market.close_time < now_utc,
        ).all()

        resolved_count = 0
        for market in pending:
            if resolve_market(market):
                resolved_count += 1
        return resolved_count
    except Exception:
        logger.exception("Failed to resolve pending markets")
        return 0


def get_resolution_summary():
    """Return aggregate summary of market/signal resolution state."""
    now_utc = datetime.now(timezone.utc)

    total_markets = Market.query.count()
    resolved_markets = Market.query.filter(Market.resolved.is_(True)).count()
    pending_resolution = Market.query.filter(
        Market.resolved.is_(False),
        Market.close_time < now_utc,
    ).count()
    unresolved_future = Market.query.filter(
        Market.resolved.is_(False),
        Market.close_time >= now_utc,
    ).count()

    latest_resolved_signal = (
        Signal.query.filter(Signal.resolved.is_(True))
        .order_by(Signal.logged_at.desc())
        .first()
    )
    last_resolved_at = _utc_iso_z(latest_resolved_signal.logged_at) if latest_resolved_signal else None

    return {
        "total_markets": total_markets,
        "resolved_markets": resolved_markets,
        "pending_resolution": pending_resolution,
        "unresolved_future": unresolved_future,
        "last_resolved_at": last_resolved_at,
    }
