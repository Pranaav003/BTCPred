"""Paper trading execution and portfolio utilities."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from app.db_helpers import get_or_create_market
from app.feature_engineering import get_live_snapshot
from app.kalshi_client import get_active_market
from app.models import AppSettings, Market, PaperTrade, Portfolio, Signal, TradeSnapshot, db

logger = logging.getLogger(__name__)


def _utc_iso_z(value: datetime | None) -> str | None:
    """Serialize datetime as UTC ISO string with trailing Z."""
    if value is None:
        return None
    if value.tzinfo is None:
        return f"{value.isoformat()}Z"
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def position_sizing_breakdown(
    p_market: float,
    side: str,
    mispricing_gap: float = 0.0,
    signal_mode: str = "agreement",
) -> dict[str, float]:
    """Return edge / mispricing / combined multipliers (combined capped at 3x) for UI and snapshots."""
    market_prob = float(p_market or 0.0)
    normalized_side = str(side or "").upper()
    if normalized_side == "YES":
        edge = 1.0 - market_prob
    else:
        edge = market_prob
    if edge >= 0.35:
        base_mult = 1.5
    elif edge >= 0.20:
        base_mult = 1.0
    elif edge >= 0.10:
        base_mult = 0.6
    else:
        base_mult = 0.3
    mode = str(signal_mode or "agreement").lower()
    gap = float(mispricing_gap or 0.0)
    if mode == "mispricing":
        if gap >= 0.40:
            mp_mult = 2.0
        elif gap >= 0.30:
            mp_mult = 1.75
        elif gap >= 0.20:
            mp_mult = 1.5
        else:
            mp_mult = 1.0
    else:
        mp_mult = 1.0
    combined = min(float(base_mult) * float(mp_mult), 3.0)
    return {
        "base_multiplier": float(base_mult),
        "mispricing_multiplier": float(mp_mult),
        "final_multiplier": float(combined),
    }


def compute_position_size(
    p_market,
    side,
    base_size,
    mispricing_gap: float = 0.0,
    signal_mode: str = "agreement",
):
    """Scale by edge, mispricing gap (mispricing mode), and cap at 40% of cash."""
    market_prob = float(p_market or 0.0)
    normalized_side = str(side or "").upper()
    base = float(base_size or 0.0)
    if base <= 0:
        return 0.0

    if normalized_side == "YES":
        edge = 1.0 - market_prob
    else:
        edge = market_prob

    if edge >= 0.35:
        base_multiplier = 1.5
    elif edge >= 0.20:
        base_multiplier = 1.0
    elif edge >= 0.10:
        base_multiplier = 0.6
    else:
        base_multiplier = 0.3

    mode = str(signal_mode or "agreement").lower()
    gap = float(mispricing_gap or 0.0)
    if mode == "mispricing":
        if gap >= 0.40:
            mispricing_multiplier = 2.0
        elif gap >= 0.30:
            mispricing_multiplier = 1.75
        elif gap >= 0.20:
            mispricing_multiplier = 1.5
        else:
            mispricing_multiplier = 1.0
    else:
        mispricing_multiplier = 1.0

    final_multiplier = min(float(base_multiplier) * float(mispricing_multiplier), 3.0)
    scaled_size = base * final_multiplier
    portfolio = Portfolio.get_or_create()
    max_size = float(portfolio.cash or 0.0) * 0.40
    final_size = min(scaled_size, max_size)
    logger.info(
        "Position sizing: base=%s, edge_mult=%sx, mispricing_mult=%sx, final_mult=%sx, final_size=$%.2f",
        base,
        base_multiplier,
        mispricing_multiplier,
        final_multiplier,
        final_size,
    )
    return max(0.0, final_size)


def _infer_mispricing_gap_for_ticker(ticker: str) -> float:
    """Use latest signal row for gap when not explicitly provided (manual trades)."""
    row = (
        Signal.query.join(Market, Signal.market_id == Market.id)
        .filter(Market.ticker == ticker)
        .order_by(Signal.logged_at.desc())
        .first()
    )
    if row is None or row.p_raw is None or row.p_market is None:
        return 0.0
    return abs(float(row.p_raw) - float(row.p_market))


def _chart_history_for_ticker(ticker: str) -> list[dict]:
    """Recent signal rows for the same market_id, oldest first (for chart at entry)."""
    market = Market.query.filter_by(ticker=ticker).first()
    if market is None:
        return []
    rows = (
        Signal.query.filter_by(market_id=market.id)
        .order_by(Signal.logged_at.desc())
        .limit(25)
        .all()
    )
    return [
        {
            "ts": s.snapshot_ts,
            "logged_at": f"{s.logged_at.isoformat()}Z" if s.logged_at else None,
            "p_market": round(float(s.p_market or 0.0), 4),
            "p_raw": round(float(s.p_raw or 0.0), 4),
        }
        for s in reversed(rows)
    ]


def _persist_trade_snapshot(
    trade: PaperTrade,
    ticker: str,
    snapshot_data: dict,
) -> None:
    if not snapshot_data:
        return
    mode = str(snapshot_data.get("signal_mode", "agreement") or "agreement").lower()
    pm = float(snapshot_data.get("p_market", 0) or 0)
    pr = float(snapshot_data.get("p_raw", 0) or 0)
    if mode == "mispricing":
        gap = abs(pr - pm)
    else:
        gap = 0.0
    reason = str(snapshot_data.get("reason", "") or "")[:256]
    chart_history = _chart_history_for_ticker(ticker)
    raw_map = snapshot_data.get("raw_features") or {}
    if not isinstance(raw_map, dict):
        raw_map = {}
    snap = TradeSnapshot(
        trade_id=trade.id,
        ticker=ticker,
        market_title=str(snapshot_data.get("market_title", "") or "")[:256],
        seconds_to_close=int(snapshot_data.get("seconds_to_close", 0) or 0),
        entry_bucket=int(snapshot_data.get("entry_bucket", 0) or 0),
        p_market=pm,
        p_raw=pr,
        signal_mode=str(snapshot_data.get("signal_mode", "unknown") or "unknown")[:32],
        agreement_region=str(snapshot_data.get("agreement_region", "") or "")[:32],
        signal_reason=reason,
        confidence=float(snapshot_data.get("confidence", 0) or 0),
        reversal_risk=float(snapshot_data.get("reversal_risk", 0) or 0),
        mispricing_gap=gap,
        btc_price=snapshot_data.get("btc_price"),
        up_price_cents=snapshot_data.get("up_price_cents"),
        down_price_cents=snapshot_data.get("down_price_cents"),
        chart_history_json=json.dumps(chart_history, default=str),
        raw_features_json=json.dumps(raw_map, default=str),
    )
    db.session.add(snap)
    db.session.commit()


def execute_paper_trade(
    side,
    ticker,
    contracts=None,
    signal_id=None,
    signal_triggered=False,
    seconds_to_close=None,
    dollar_amount=None,
    use_dynamic_sizing=False,
    snapshot_data=None,
    mispricing_gap=None,
    signal_mode=None,
):
    """Execute a paper trade inside Flask app context."""
    normalized_side = str(side or "").upper()
    if normalized_side not in {"YES", "NO"}:
        return {"error": "Invalid side. Must be YES or NO."}

    if seconds_to_close is not None and int(seconds_to_close) < 60:
        return {
            "error": "Too close to expiry",
            "detail": "Kalshi locks trading within 60s of close",
        }

    portfolio = Portfolio.get_or_create()

    ticker = str(ticker or "").strip()
    if not ticker:
        return {"error": "Ticker is required."}

    if signal_triggered:
        if PaperTrade.has_recent_auto_trade(ticker, normalized_side, minutes=20):
            return {
                "error": f"Auto-trade already placed for {ticker} {normalized_side} within the last 20 minutes",
            }

    p_market = None
    active_market = get_active_market()
    if active_market and active_market.get("ticker") == ticker:
        snapshot = get_live_snapshot()
        if snapshot and snapshot.get("market_ticker") == ticker:
            p_market = snapshot.get("p_market")

    existing_market = Market.query.filter_by(ticker=ticker).first()
    if p_market is None:
        latest_signal = (
            Signal.query.join(Market, Signal.market_id == Market.id)
            .filter(Market.ticker == ticker)
            .order_by(Signal.logged_at.desc())
            .first()
        )
        if latest_signal is not None:
            p_market = latest_signal.p_market

    if p_market is None:
        return {"error": "Unable to determine current market price for ticker."}

    p_market = float(p_market)
    if normalized_side == "YES":
        entry_price = p_market
    else:
        # NO contracts are priced as the complement of YES.
        entry_price = 1.0 - p_market
    if entry_price <= 0:
        return {"error": "Invalid entry price for trade."}

    if dollar_amount is not None:
        base_size = float(dollar_amount or 0.0)
        if base_size <= 0:
            return {"error": "Dollar amount must be greater than zero."}
        if use_dynamic_sizing:
            mode = (signal_mode or AppSettings.get("signal_mode", "agreement") or "agreement").lower()
            mg: float
            if mispricing_gap is not None:
                mg = float(mispricing_gap)
            elif mode == "mispricing":
                live = get_live_snapshot()
                if live and str(live.get("market_ticker", "")) == ticker:
                    mg = abs(float(live.get("p_raw", 0) or 0) - float(live.get("p_market", 0) or 0))
                else:
                    mg = _infer_mispricing_gap_for_ticker(ticker)
            else:
                mg = 0.0
            effective_size = compute_position_size(
                p_market=p_market,
                side=normalized_side,
                base_size=base_size,
                mispricing_gap=mg,
                signal_mode=mode,
            )
        else:
            effective_size = base_size
        contracts_value = effective_size / entry_price if entry_price > 0 else 0.0
    else:
        contracts_value = float(contracts or 0.0)
        effective_size = contracts_value * entry_price

    if contracts_value <= 0:
        return {"error": "Contracts must be greater than zero."}

    entry_cost = contracts_value * entry_price

    if float(portfolio.cash or 0.0) < entry_cost:
        return {"error": "Insufficient funds", "cash": float(portfolio.cash or 0.0), "required": entry_cost}

    portfolio.cash = float(portfolio.cash or 0.0) - entry_cost

    market_ref = existing_market
    if market_ref is None:
        if active_market and active_market.get("ticker") == ticker and active_market.get("close_ts") is not None:
            market_ref = get_or_create_market(
                ticker=ticker,
                title=active_market.get("title"),
                close_time=datetime.fromtimestamp(int(active_market["close_ts"]), tz=timezone.utc),
                series_ticker=active_market.get("series_ticker", "KXBTC15M"),
            )
        else:
            return {"error": "Ticker not found in database and not active on Kalshi."}

    trade = PaperTrade(
        portfolio_id=portfolio.id,
        market_id=market_ref.id if market_ref else None,
        ticker=ticker,
        side=normalized_side,
        contracts=contracts_value,
        entry_price=entry_price,
        entry_cost=entry_cost,
        signal_triggered=bool(signal_triggered),
        signal_id=signal_id,
    )
    db.session.add(trade)
    db.session.commit()

    if snapshot_data and isinstance(snapshot_data, dict):
        try:
            _persist_trade_snapshot(trade, ticker, snapshot_data)
        except Exception:
            logger.exception("Failed to persist trade snapshot for trade_id=%s", trade.id)

    return {
        "success": True,
        "trade_id": trade.id,
        "entry_price": entry_price,
        "entry_cost": entry_cost,
        "effective_size": effective_size,
        "dynamic_sizing_used": bool(use_dynamic_sizing and dollar_amount is not None),
        "cash_remaining": float(portfolio.cash or 0.0),
    }


def get_portfolio_summary():
    """Return aggregate paper portfolio summary metrics."""
    portfolio = Portfolio.get_or_create()

    realized_query = PaperTrade.query.filter(PaperTrade.resolved.is_(True))
    open_query = PaperTrade.query.filter(PaperTrade.resolved.is_(False))

    realized_pnl = float(sum((row.realized_pnl or 0.0) for row in realized_query.all()))
    open_cost = float(sum((row.entry_cost or 0.0) for row in open_query.all()))
    unrealized_pnl = 0.0
    cash_value = float(portfolio.cash or 0.0)
    if cash_value < 0:
        logger.error("Portfolio cash negative (%.4f). Clamping to 0.0 to prevent invalid accounting.", cash_value)
        cash_value = 0.0
        portfolio.cash = 0.0
        db.session.commit()
    total_value = float(cash_value + open_cost)

    resolved_count = realized_query.count()
    wins = realized_query.filter(PaperTrade.outcome_correct.is_(True)).count()
    win_rate = (wins / resolved_count) if resolved_count else None

    base = float(portfolio.starting_balance or 0.0)
    total_return_pct = ((cash_value - base) / base * 100.0) if base else 0.0

    return {
        "cash": cash_value,
        "starting_balance": float(portfolio.starting_balance or 0.0),
        "realized_pnl": realized_pnl,
        "unrealized_pnl": unrealized_pnl,
        "total_value": total_value,
        "total_return_pct": total_return_pct,
        "open_trades": open_query.count(),
        "total_trades": PaperTrade.query.count(),
        "win_rate": win_rate,
    }


def get_open_positions():
    """Return unresolved paper trades for current open positions."""
    rows = PaperTrade.query.filter(PaperTrade.resolved.is_(False)).order_by(PaperTrade.entry_at.desc()).all()
    return [
        {
            "id": row.id,
            "ticker": row.ticker,
            "side": row.side,
            "contracts": row.contracts,
            "entry_price": row.entry_price,
            "entry_cost": row.entry_cost,
            "entry_at": _utc_iso_z(row.entry_at),
            "signal_triggered": row.signal_triggered,
            "current_value": row.current_value,
            "unrealized_pnl": row.unrealized_pnl,
        }
        for row in rows
    ]


def get_trade_history(limit=50):
    """Return resolved paper trades ordered by newest exit first."""
    rows = (
        PaperTrade.query.filter(PaperTrade.resolved.is_(True))
        .order_by(PaperTrade.exit_at.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id": row.id,
            "ticker": row.ticker,
            "side": row.side,
            "contracts": row.contracts,
            "entry_price": row.entry_price,
            "exit_price": row.exit_price,
            "entry_cost": row.entry_cost,
            "realized_pnl": row.realized_pnl,
            "pnl_display": row.pnl_display,
            "outcome_correct": row.outcome_correct,
            "entry_at": _utc_iso_z(row.entry_at),
            "exit_at": _utc_iso_z(row.exit_at),
            "signal_triggered": row.signal_triggered,
        }
        for row in rows
    ]


def reset_portfolio(starting_balance=100.0):
    """Reset paper portfolio and wipe all paper trades."""
    balance = float(starting_balance or 100.0)
    TradeSnapshot.query.delete()
    PaperTrade.query.delete()
    portfolio = Portfolio.get_or_create()
    portfolio.cash = balance
    portfolio.starting_balance = balance
    portfolio.total_deposited = balance
    db.session.commit()
    return get_portfolio_summary()
