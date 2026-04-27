"""Background polling scheduler for live signal generation and persistence."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from apscheduler.schedulers.background import BackgroundScheduler

from app.db_helpers import export_training_data, get_or_create_market, save_signal
from app.feature_engineering import get_live_snapshot
from app.kalshi_client import get_active_market, get_btc_price, get_market_prices
from app.models import AppSettings, PaperTrade
from app.paper_trading import execute_paper_trade
from app.resolver import resolve_pending_markets
from app.signal_engine import evaluate_live_signal, signal_to_dict
from train_raw_model import RAW_FEATURES

logger = logging.getLogger(__name__)
_SCHEDULER_INSTANCE: BackgroundScheduler | None = None
_app = None
_latest_snapshot = None
_latest_signal = None
MIN_SECONDS_FOR_AUTO_TRADE = 90

_SNAPSHOT_FEATURE_KEYS = [
    "return_1m",
    "return_3m",
    "return_5m",
    "volatility_3m",
    "volatility_5m",
    "momentum_1m",
    "momentum_3m",
    "trade_count_1m",
    "volume_1m",
    "flip_count_5m",
    "reversal_risk",
]


def _price_context_for_snapshot() -> dict:
    """BTC and Kalshi YES/NO ask prices in cents for trade snapshots."""
    out: dict = {"btc_price": None, "up_price_cents": None, "down_price_cents": None}
    out["btc_price"] = get_btc_price()
    market = get_active_market()
    ticker = market.get("ticker") if isinstance(market, dict) else None
    if not ticker:
        return out
    quote = get_market_prices(ticker)
    if not quote:
        return out

    def _cents(value):
        if value is None:
            return None
        try:
            return int(round(float(value) * 100))
        except (TypeError, ValueError):
            return None

    out["up_price_cents"] = _cents(quote.get("yes_ask"))
    out["down_price_cents"] = _cents(quote.get("no_ask"))
    return out


def poll_and_signal() -> None:
    """Scheduled polling job that computes and stores the latest signal."""
    try:
        if _app is None:
            logger.error("poll_and_signal called before scheduler app initialization.")
            return

        with _app.app_context():
            if AppSettings.get("scheduler_running", "false") == "false":
                logger.debug("Scheduler paused")
                return

            global _latest_snapshot, _latest_signal
            snapshot = get_live_snapshot()
            if snapshot is None:
                logger.warning("No live snapshot available; skipping poll cycle.")
                return
            logger.debug("Live snapshot received with keys: %s", sorted(snapshot.keys()))

            result = evaluate_live_signal(snapshot)
            if result is None:
                logger.warning("Live signal evaluation unavailable; model may not be loaded.")
                return
            logger.debug("Signal evaluation result: %s", signal_to_dict(result))
            _latest_snapshot = snapshot
            _latest_signal = signal_to_dict(result)

            market = get_or_create_market(
                ticker=str(snapshot["market_ticker"]),
                title=snapshot.get("market_title"),
                close_time=datetime.fromtimestamp(int(snapshot["close_ts"]), tz=timezone.utc),
                series_ticker="KXBTC15M",
            )

            raw_features_dict = {key: snapshot.get(key, 0.0) for key in RAW_FEATURES}
            saved_signal = save_signal(
                market=market,
                snapshot_dict=snapshot,
                signal_str=result.signal,
                reason_str=result.reason,
                agreement_region_str=result.agreement_region,
                p_raw=result.p_raw,
                yes_cutoff=result.yes_cutoff,
                no_cutoff=result.no_cutoff,
                raw_features_dict=raw_features_dict,
            )
            logger.info("Signal saved to DB, id=%s", saved_signal.id)

            auto_trade_enabled = AppSettings.get("auto_trade_enabled", "false") == "true"
            paper_trading_enabled = AppSettings.get("paper_trading_enabled", "false") == "true"
            if auto_trade_enabled and paper_trading_enabled and result.signal in ("PAPER BUY YES", "PAPER BUY NO"):
                if result is None:
                    logger.error("Signal result is None, skipping auto-trade")
                    return
                if result.p_market <= 0 or result.p_market >= 1:
                    logger.error("Invalid p_market=%s, skipping", result.p_market)
                    return
                if result.p_raw <= 0 or result.p_raw >= 1:
                    logger.error("Invalid p_raw=%s, skipping", result.p_raw)
                    return
                side = "YES" if result.signal == "PAPER BUY YES" else "NO"
                if PaperTrade.has_recent_auto_trade(str(snapshot["market_ticker"]), side, minutes=20):
                    logger.debug(
                        "Dedup: already placed %s auto-trade for %s in the last 20m, skipping",
                        side,
                        snapshot["market_ticker"],
                    )
                else:
                    seconds_left = int(snapshot.get("seconds_to_close", 0) or 0)
                    if seconds_left < MIN_SECONDS_FOR_AUTO_TRADE:
                        logger.info(
                            "Auto-trade skipped: only %ss to close, minimum is %ss",
                            seconds_left,
                            MIN_SECONDS_FOR_AUTO_TRADE,
                        )
                    else:
                        reversal_risk = float(snapshot.get("reversal_risk", 0.0) or 0.0)
                        max_reversal = float(AppSettings.get("max_reversal_risk", "0.65") or 0.65)
                        if reversal_risk > max_reversal:
                            logger.info(
                                "Auto-trade blocked by volatility guard: reversal_risk=%.3f > max=%.3f",
                                reversal_risk,
                                max_reversal,
                            )
                            return
                        dynamic_sizing = AppSettings.get("dynamic_sizing_enabled", "false") == "true"
                        dollar_amount = float(AppSettings.get("paper_trade_size", "10.0"))
                        mode_str = (AppSettings.get("signal_mode", "agreement") or "agreement").lower()
                        mispricing_gap = (
                            abs(float(result.p_raw) - float(result.p_market))
                            if mode_str == "mispricing"
                            else 0.0
                        )

                        entry_price = float(result.p_market or 0.0) if side == "YES" else (1.0 - float(result.p_market or 0.0))
                        contracts = (dollar_amount / entry_price) if entry_price > 0 else 0.0
                        if contracts > 0:
                            price_ctx = _price_context_for_snapshot()
                            snapshot_data = {
                                "market_title": snapshot.get("market_title", "") or "",
                                "seconds_to_close": int(snapshot.get("seconds_to_close", 0) or 0),
                                "entry_bucket": int(snapshot.get("entry_bucket", 0) or 0),
                                "p_market": float(snapshot.get("p_market", 0) or 0),
                                "p_raw": float(snapshot.get("p_raw", 0) or 0),
                                "signal_mode": AppSettings.get("signal_mode", "agreement") or "agreement",
                                "agreement_region": result.agreement_region,
                                "reason": result.reason,
                                "confidence": float(result.confidence or 0),
                                "reversal_risk": float(snapshot.get("reversal_risk", 0) or 0),
                                "raw_features": {k: snapshot.get(k) for k in _SNAPSHOT_FEATURE_KEYS},
                                **price_ctx,
                            }
                            trade_result = execute_paper_trade(
                                side=side,
                                contracts=contracts,
                                dollar_amount=dollar_amount,
                                use_dynamic_sizing=dynamic_sizing,
                                ticker=str(snapshot["market_ticker"]),
                                signal_id=saved_signal.id,
                                signal_triggered=True,
                                seconds_to_close=result.seconds_to_close,
                                snapshot_data=snapshot_data,
                                mispricing_gap=mispricing_gap,
                                signal_mode=mode_str,
                            )
                            logger.info(
                                "Auto-trade executed: %s %.2f contracts on %s | result=%s",
                                side,
                                contracts,
                                snapshot["market_ticker"],
                                trade_result,
                            )
                        else:
                            logger.warning("Skipped auto-trade because entry price is invalid for side %s", side)

            logger.info(
                "Signal: %s | p_market=%.3f | p_raw=%.3f | %s",
                result.signal,
                result.p_market,
                result.p_raw,
                result.reason,
            )
    except Exception:
        logger.exception("poll_and_signal failed")


def resolve_job() -> None:
    """Scheduled resolution job to settle recently closed markets."""
    try:
        if _app is None:
            logger.error("resolve_job called before scheduler app initialization.")
            return
        with _app.app_context():
            count = resolve_pending_markets()
            logger.info("Resolution check complete, %s markets resolved", count)
    except Exception:
        logger.exception("resolve_job failed")


def auto_export_job() -> None:
    """Scheduled export of live training-ready CSV every 6 hours."""
    try:
        if _app is None:
            logger.error("auto_export_job called before scheduler app initialization.")
            return
        with _app.app_context():
            rows, skipped = export_training_data("live_training_data.csv")
            logger.info(
                "Auto-exported %s training rows (%s skipped) to live_training_data.csv",
                rows,
                skipped,
            )
    except Exception:
        logger.exception("auto_export_job failed")


def get_latest_snapshot():
    """Return latest cached market snapshot from scheduler polling."""
    return _latest_snapshot


def get_latest_signal():
    """Return latest cached signal payload from scheduler polling."""
    return _latest_signal


def init_scheduler(app):
    """Initialize and start background scheduler with polling job."""
    global _SCHEDULER_INSTANCE, _app

    _app = app

    if _SCHEDULER_INSTANCE is not None and _SCHEDULER_INSTANCE.running:
        return _SCHEDULER_INSTANCE

    with app.app_context():
        interval = int(AppSettings.get("poll_interval_seconds", "15"))

    scheduler_instance = BackgroundScheduler(timezone="UTC")
    scheduler_instance.add_job(
        poll_and_signal,
        trigger="interval",
        seconds=interval,
        id="poll_signal",
        replace_existing=True,
        misfire_grace_time=10,
    )
    scheduler_instance.add_job(
        resolve_job,
        trigger="interval",
        seconds=60,
        id="resolve_markets",
        replace_existing=True,
        misfire_grace_time=30,
    )
    scheduler_instance.add_job(
        auto_export_job,
        trigger="interval",
        hours=6,
        id="auto_export",
        replace_existing=True,
        misfire_grace_time=60,
    )
    scheduler_instance.start()
    logger.info("Scheduler started, polling every %ss", interval)

    _SCHEDULER_INSTANCE = scheduler_instance
    return scheduler_instance
