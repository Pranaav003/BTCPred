"""Database helper utilities for markets, signals, and app settings."""

from __future__ import annotations

import json
import logging
import csv
from datetime import datetime, timezone

from sqlalchemy import func
from sqlalchemy.orm import contains_eager

from app.models import AppSettings, Market, PaperTrade, Portfolio, Signal, db

logger = logging.getLogger(__name__)


def _utc_iso_z(value: datetime | None) -> str | None:
    """Serialize datetime as UTC ISO string with trailing Z."""
    if value is None:
        return None
    if value.tzinfo is None:
        return f"{value.isoformat()}Z"
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def get_or_create_market(
    ticker: str,
    title: str | None,
    close_time: datetime,
    series_ticker: str | None,
) -> Market:
    """Fetch existing market by ticker or create it."""
    market = Market.query.filter_by(ticker=ticker).first()
    if market is not None:
        return market

    market = Market(
        ticker=ticker,
        title=title,
        close_time=close_time,
        series_ticker=series_ticker or "KXBTC15M",
    )
    db.session.add(market)
    db.session.commit()
    return market


def save_signal(
    market: Market,
    snapshot_dict: dict,
    signal_str: str,
    reason_str: str | None,
    agreement_region_str: str | None,
    p_raw: float | None,
    yes_cutoff: float | None,
    no_cutoff: float | None,
    raw_features_dict: dict,
) -> Signal:
    """Persist a signal row for the given market snapshot."""
    signal = Signal(
        market_id=market.id,
        snapshot_ts=snapshot_dict.get("snapshot_ts"),
        seconds_to_close=snapshot_dict.get("seconds_to_close"),
        entry_bucket=snapshot_dict.get("entry_bucket"),
        p_market=snapshot_dict.get("p_market"),
        p_raw=p_raw,
        orderbook_mid=None,
        orderbook_available=False,
        yes_cutoff=yes_cutoff,
        no_cutoff=no_cutoff,
        signal=signal_str,
        reason=reason_str,
        agreement_region=agreement_region_str,
        raw_features_json=json.dumps(raw_features_dict or {}),
    )
    db.session.add(signal)
    db.session.commit()
    return signal


def get_recent_signals(limit: int = 50) -> list[dict]:
    """Return recent signal rows joined with market metadata."""
    rows = (
        db.session.query(Signal, Market)
        .join(Market, Signal.market_id == Market.id)
        .order_by(Signal.logged_at.desc())
        .limit(limit)
        .all()
    )

    def _format_pnl(value):
        if value is None:
            return "--"
        return f"{value:+.3f}"

    return [
        {
            "id": signal.id,
            "logged_at": _utc_iso_z(signal.logged_at),
            "ticker": market.ticker,
            "title": market.title,
            "close_time": _utc_iso_z(market.close_time),
            "seconds_to_close": signal.seconds_to_close,
            "entry_bucket": signal.entry_bucket,
            "p_market": signal.p_market,
            "p_raw": signal.p_raw,
            "signal": signal.signal,
            "reason": signal.reason,
            "agreement_region": signal.agreement_region,
            "resolved": signal.resolved,
            "pnl": _format_pnl(signal.pnl),
            "pnl_raw": signal.pnl,
            "outcome_correct": signal.outcome_correct,
            "outcome_yes": market.final_outcome_yes,
            "resolution_price": market.resolution_price,
        }
        for signal, market in rows
    ]


def get_signal_metrics() -> dict:
    """Return aggregate signal metrics for non-NO-SIGNAL rows."""
    base_query = Signal.query.filter(Signal.signal != "NO SIGNAL")

    total_signals = base_query.count()
    yes_signals = base_query.filter(Signal.signal == "PAPER BUY YES").count()
    no_signals = base_query.filter(Signal.signal == "PAPER BUY NO").count()

    resolved_query = base_query.filter(Signal.resolved.is_(True))
    resolved_count = resolved_query.count()
    correct_count = resolved_query.filter(Signal.outcome_correct.is_(True)).count()
    accuracy = (correct_count / resolved_count) if resolved_count else None

    pnl_aggregates = resolved_query.with_entities(
        func.avg(Signal.pnl),
        func.sum(Signal.pnl),
        func.max(Signal.pnl),
        func.min(Signal.pnl),
    ).first()

    avg_pnl, total_pnl, best_pnl, worst_pnl = pnl_aggregates if pnl_aggregates else (None, None, None, None)
    today_date = datetime.now(timezone.utc).date()
    entry_filtered_total = Signal.query.filter(Signal.agreement_region == "entry_filtered").count()
    entry_filtered_today = Signal.query.filter(
        Signal.agreement_region == "entry_filtered",
        func.date(Signal.logged_at) == today_date.isoformat(),
    ).count()
    volatility_guard_today = Signal.query.filter(
        Signal.agreement_region == "volatility_guard",
        func.date(Signal.logged_at) == today_date.isoformat(),
    ).count()
    outside_time_window_today = Signal.query.filter(
        Signal.agreement_region == "outside_time_window",
        func.date(Signal.logged_at) == today_date.isoformat(),
    ).count()

    return {
        "total_signals": total_signals,
        "yes_signals": yes_signals,
        "no_signals": no_signals,
        "resolved_count": resolved_count,
        "accuracy": accuracy,
        "avg_pnl": float(avg_pnl) if avg_pnl is not None else None,
        "total_pnl": float(total_pnl) if total_pnl is not None else None,
        "best_pnl": float(best_pnl) if best_pnl is not None else None,
        "worst_pnl": float(worst_pnl) if worst_pnl is not None else None,
        "entry_filtered_today": int(entry_filtered_today),
        "entry_filtered_total": int(entry_filtered_total),
        "volatility_guard_today": int(volatility_guard_today),
        "outside_time_window_today": int(outside_time_window_today),
    }


def get_probability_history(limit: int = 50) -> list[dict]:
    """Return ascending probability history for charting."""
    rows = (
        db.session.query(
            Signal.logged_at,
            Signal.p_market,
            Signal.p_raw,
            Signal.signal,
            Market.title,
            Market.close_time,
        )
        .join(Market, Signal.market_id == Market.id)
        .order_by(Signal.logged_at.desc())
        .limit(limit)
        .all()
    )
    rows.reverse()

    return [
        {
            "logged_at": _utc_iso_z(logged_at),
            "p_market": p_market,
            "p_raw": p_raw,
            "signal": sig,
            "market_title": title,
            "close_time_iso": _utc_iso_z(close_time),
        }
        for logged_at, p_market, p_raw, sig, title, close_time in rows
    ]


def seed_default_settings() -> None:
    """Seed default app settings only when key is missing."""
    defaults = {
        "yes_cutoff": "0.65",
        "no_cutoff": "0.35",
        "min_seconds_to_close": "30",
        "max_seconds_to_close": "300",
        "poll_interval_seconds": "30",
        "enable_no_signals": "false",
        "auto_trade_enabled": "false",
        "paper_trading_enabled": "false",
        "paper_trade_size": "10.0",
        "dynamic_sizing_enabled": "false",
        "risk_profile": "moderate",
        "signal_mode": "ensemble",
        "mispricing_threshold": "0.10",
        "max_entry_price_yes": "0.80",
        "max_entry_price_no": "0.80",
        "min_expected_profit": "0.10",
        "max_reversal_risk": "0.65",
        "max_daily_loss": "50.0",
        "high_conviction_volatility_override": "0.80",
        "scheduler_running": "false",
        "live_trading_enabled": "false",
        "live_trade_size": "5.0",
        "cutoff_buffer": "0.03",
        "max_mispricing_override_risk": "0.65",
        "min_auto_trade_confidence": "0.35",
    }

    for key, value in defaults.items():
        if AppSettings.get(key) is None:
            AppSettings.set(key, value)


def resolve_paper_trades(market: Market) -> int:
    """Resolve all unresolved paper trades for a market ticker."""
    trades = PaperTrade.query.filter_by(ticker=market.ticker, resolved=False).all()
    if not trades:
        return 0

    portfolio = Portfolio.get_or_create()
    outcome_yes = bool(market.final_outcome_yes)
    now_utc = datetime.now(timezone.utc)

    resolved_count = 0
    for trade in trades:
        side = (trade.side or "").upper()
        contracts = float(trade.contracts or 0.0)
        entry_price = float(trade.entry_price or 0.0)
        yes_wins = bool(outcome_yes)
        if side == "YES":
            exit_price = 1.0 if yes_wins else 0.0
            win = yes_wins
        else:
            exit_price = 1.0 if not yes_wins else 0.0
            win = not yes_wins

        realized_pnl = (exit_price - entry_price) * contracts
        cash_returned = contracts * 1.0 if exit_price > 0 else 0.0

        trade.exit_price = exit_price
        trade.exit_at = now_utc
        trade.realized_pnl = realized_pnl
        trade.outcome_correct = win
        trade.resolved = True

        if cash_returned > 0:
            # Winning YES and winning NO both pay $1.00 per contract.
            portfolio.cash = float(portfolio.cash or 0.0) + cash_returned
        logger.info(
            "Resolved trade %s: %s on %s, pnl=%.4f, cash_returned=%.4f, new_portfolio_cash=%.4f",
            trade.id,
            trade.side,
            trade.ticker,
            realized_pnl,
            cash_returned,
            float(portfolio.cash or 0.0),
        )
        resolved_count += 1

    db.session.commit()
    return resolved_count


def export_training_data(output_path: str = "live_training_data.csv") -> tuple[int, int]:
    """
    Export resolved signal rows into training-ready CSV.

    Streams one row at a time to ``output_path`` (no giant in-memory list)
    so large exports do not exhaust RAM on small workers.

    Returns:
      (rows_written, rows_skipped)
    """
    raw_features = [
        "seconds_to_close", "entry_bucket",
        "return_1m", "return_3m", "return_5m",
        "volatility_3m", "volatility_5m", "range_5m",
        "abs_return_1m", "trade_count_1m", "trade_count_3m",
        "trade_count_5m", "volume_1m", "volume_3m", "volume_5m",
        "avg_trade_price_1m", "avg_trade_price_3m",
        "momentum_acceleration",
        "flip_count_5m", "return_1m_x_inv_time",
        "return_3m_x_inv_time", "volatility_5m_x_inv_time",
        "bid_ask_spread", "rsi_14", "session",
        "distance_from_strike", "outcome_rate_bucket",
        "return_5m_ratio",
        "was_missing_return_1m", "was_missing_return_3m",
        "was_missing_return_5m", "was_missing_volatility_3m",
        "was_missing_volatility_5m",
    ]
    fieldnames = (
        raw_features
        + ["price_now", "final_outcome_yes"]
        + ["logged_at", "market_ticker", "close_ts", "p_raw", "agreement_region", "signal", "source"]
    )

    query = (
        Signal.query.join(Signal.market)
        .filter(
            Signal.resolved.is_(True),
            Signal.raw_features_json.isnot(None),
            Market.final_outcome_yes.isnot(None),
        )
        .options(contains_eager(Signal.market))
        .order_by(Signal.logged_at.asc())
    )

    rows_written = 0
    skipped = 0

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in query.yield_per(250):
            try:
                features = json.loads(s.raw_features_json or "{}")
            except Exception:
                skipped += 1
                continue
            market = s.market
            if market is None or market.final_outcome_yes is None:
                skipped += 1
                continue
            row = {feature: features.get(feature, 0.0) for feature in raw_features}
            row["price_now"] = s.p_market
            row["final_outcome_yes"] = int(market.final_outcome_yes)
            row["logged_at"] = _utc_iso_z(s.logged_at)
            row["market_ticker"] = market.ticker
            row["close_ts"] = int(market.close_time.timestamp()) if market.close_time is not None else None
            row["p_raw"] = s.p_raw
            row["agreement_region"] = s.agreement_region
            row["signal"] = s.signal
            row["source"] = "live"
            writer.writerow(row)
            rows_written += 1

    return rows_written, skipped
