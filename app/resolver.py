"""Market resolution and signal PnL backfill utilities."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from app.extensions import db
from app.db_helpers import get_setting, resolve_paper_trades, set_setting
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
        logger.warning(
            "Market %s finalized without yes/no result — skipping resolution",
            market.ticker,
        )
        return False

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
    """Resolve all past-close unresolved markets (last 24h only to avoid API spam).

    Older markets that Kalshi returns 500 for are marked resolved to stop
    the resolver from retrying them every 60 seconds and eating rate limit budget.
    """
    from datetime import timedelta

    try:
        now_utc = datetime.now(timezone.utc)
        # Only attempt resolution for markets closed in the last 24 hours.
        # Older markets consistently return 500 from Kalshi and just waste API calls.
        cutoff = now_utc - timedelta(hours=24)
        pending = Market.query.filter(
            Market.resolved.is_(False),
            Market.close_time < now_utc,
            Market.close_time >= cutoff,
        ).all()

        resolved_count = 0
        for market in pending:
            if resolve_market(market):
                resolved_count += 1

        # Mark very old unresolved markets as resolved (unknown outcome) to stop retrying.
        ancient = Market.query.filter(
            Market.resolved.is_(False),
            Market.close_time < cutoff,
        ).all()
        if ancient:
            for market in ancient:
                market.resolved = True
            db.session.commit()
            logger.info("Marked %d ancient unresolved markets as resolved (stale)", len(ancient))

        return resolved_count
    except Exception:
        logger.exception("Failed to resolve pending markets")
        return 0


def _apply_kalshi_settlement(trade, settlement: dict, now_utc: datetime) -> bool:
    """Update a live trade from Kalshi /portfolio/settlements (source of truth)."""
    from app.kalshi_trader import _parse_fp_count, _parse_fp_dollars

    market_result = str(settlement.get("market_result") or "").strip().lower()
    if market_result not in {"yes", "no"}:
        return False

    trade_is_yes = trade.side.upper() == "YES"
    yes_count = _parse_fp_count(settlement.get("yes_count_fp"))
    no_count = _parse_fp_count(settlement.get("no_count_fp"))
    side_count = yes_count if trade_is_yes else no_count

    if side_count <= 0:
        trade.resolved = True
        trade.order_status = "unfilled"
        trade.contracts = 0.0
        trade.cost_dollars = 0.0
        trade.exit_price = 0.0
        trade.realized_pnl = 0.0
        trade.outcome = "unfilled"
        trade.resolved_at = now_utc
        return True

    yes_cost = float(_parse_fp_dollars(settlement.get("yes_total_cost_dollars")) or 0.0)
    no_cost = float(_parse_fp_dollars(settlement.get("no_total_cost_dollars")) or 0.0)
    fee_cost = float(_parse_fp_dollars(settlement.get("fee_cost")) or 0.0)
    revenue_cents = int(settlement.get("revenue") or 0)
    side_cost = yes_cost if trade_is_yes else no_cost
    pnl = round((revenue_cents / 100.0) - side_cost - fee_cost, 2)

    outcome_yes = market_result == "yes"
    won = (trade_is_yes and outcome_yes) or (not trade_is_yes and not outcome_yes)

    trade.resolved = True
    trade.order_status = "placed"
    trade.contracts = side_count
    trade.cost_dollars = round(side_cost, 2)
    trade.exit_price = 1.0 if won else 0.0
    trade.realized_pnl = pnl
    trade.outcome = "correct" if won else "wrong"
    trade.resolved_at = now_utc
    return True


def resolve_live_trades() -> int:
    """Resolve or reconcile live trades using Kalshi settlement data.

    Only processes trades from the last 48 hours to avoid hammering the API
    with old tickers that return 500.
    """
    from datetime import timedelta
    from app.kalshi_trader import cancel_order, get_settlement_for_ticker, is_configured
    from app.models import LiveTrade, Market

    if not is_configured():
        return 0

    # Cancel resting GTC orders ONLY on markets that have already closed.
    # Active-market resting orders should be left alone — they may still fill.
    now_utc = datetime.now(timezone.utc)
    closed_tickers = {
        m.ticker
        for m in Market.query.filter(Market.close_time < now_utc).all()
    }
    resting_trades = LiveTrade.query.filter(
        LiveTrade.order_status == "resting",
        LiveTrade.kalshi_order_id.isnot(None),
    ).all()
    cancelled_count = 0
    for rt in resting_trades:
        if rt.ticker in closed_tickers:
            logger.info(
                "Cancelling resting GTC order %s on closed market %s during resolution",
                rt.kalshi_order_id,
                rt.ticker,
            )
            cancel_order(rt.kalshi_order_id)
            rt.order_status = "cancelled"
            # Clear the resting-order tracker so the scheduler doesn't try
            # to query a dead order on the next poll.
            set_setting(f"live_resting_order_{rt.ticker}", "")
            cancelled_count += 1
        else:
            logger.debug(
                "Leaving resting GTC order %s on active market %s (waiting for fill)",
                rt.kalshi_order_id,
                rt.ticker,
            )
    if cancelled_count:
        from app import db as _db
        _db.session.commit()

    # Only try to settle trades from last 48h — older ones just spam 500s.
    cutoff = now_utc - timedelta(hours=48)
    trades = (
        LiveTrade.query.filter(
            LiveTrade.kalshi_order_id.isnot(None),
            LiveTrade.order_status != "failed",
            LiveTrade.entry_at >= cutoff,
        )
        .order_by(LiveTrade.entry_at.asc())
        .all()
    )
    if not trades:
        return 0

    updated = 0
    settlement_cache: dict[str, dict | None] = {}
    resolution_cache: dict[str, dict | None] = {}

    for trade in trades:
        ticker = trade.ticker
        if ticker not in settlement_cache:
            settlement_cache[ticker] = get_settlement_for_ticker(ticker)
        settlement = settlement_cache[ticker]
        if settlement is not None:
            if _apply_kalshi_settlement(trade, settlement, now_utc):
                updated += 1
            continue

        if ticker not in resolution_cache:
            resolution_cache[ticker] = get_market_resolution(ticker)
        resolution = resolution_cache[ticker]
        if not resolution or resolution.get("resolved") is not True:
            continue
        if resolution.get("result") not in ("yes", "no"):
            continue

        trade.resolved = True
        trade.order_status = "unfilled"
        trade.contracts = 0.0
        trade.cost_dollars = 0.0
        trade.exit_price = 0.0
        trade.realized_pnl = 0.0
        trade.outcome = "unfilled"
        trade.resolved_at = now_utc
        updated += 1

    if updated:
        db.session.commit()
    return updated


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
