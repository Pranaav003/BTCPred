"""Background polling scheduler for live signal generation and persistence."""

from __future__ import annotations

import logging
import math
import threading
import time
from datetime import datetime, timezone

from apscheduler.schedulers.background import BackgroundScheduler

from app.db_helpers import export_training_data, get_or_create_market, get_setting, save_signal, set_setting
from app.feature_engineering import get_live_snapshot
from app.kalshi_client import get_active_market, get_btc_price, get_market_prices
from app.models import PaperTrade
from app.paper_trading import execute_paper_trade, get_realized_pnl_today_utc
from app.kalshi_auth import is_configured as kalshi_configured
from app.resolver import resolve_pending_markets, resolve_live_trades
from app.signal_engine import evaluate_live_signal, signal_to_dict
from train_raw_model import RAW_FEATURES

logger = logging.getLogger(__name__)
_SCHEDULER_INSTANCE: BackgroundScheduler | None = None
_app = None
_latest_snapshot = None
_latest_signal = None
MIN_SECONDS_FOR_AUTO_TRADE = 90
_consecutive_failures = 0
_MAX_FAILURES_BEFORE_COOLDOWN = 3
_COOLDOWN_SECONDS = 60
_cooldown_until_ts = 0.0
_cooldown_level = 0
_COOLDOWN_SCHEDULE = [60, 120, 240, 300]


def _auto_trade_allowed_by_daily_loss() -> bool:
    """False when net realized PnL from exits today (UTC) is at or below -max_daily_loss."""
    max_daily_loss = float(get_setting("max_daily_loss", "50.0") or 50.0)
    if max_daily_loss <= 0:
        return True
    today_pnl = get_realized_pnl_today_utc()
    if today_pnl <= -max_daily_loss:
        logger.warning(
            "Daily loss limit reached: net realized today (UTC) %.2f <= -%.2f; skipping auto-trade.",
            today_pnl,
            max_daily_loss,
        )
        return False
    return True

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


def _execute_live_trade(result, snapshot, saved_signal, app) -> None:
    """Place a real Kalshi order mirroring the signal. All safety checks run first."""
    from datetime import datetime, timezone

    from app.kalshi_trader import get_balance, place_order
    from app.model_loader import get_model
    from app.models import LiveTrade, db
    from app.signal_engine import MIN_ENTRY_PRICE

    with app.app_context():
        try:
            today = datetime.now(timezone.utc).date()
            today_start = datetime.combine(today, datetime.min.time(), tzinfo=timezone.utc)
            today_trades = LiveTrade.query.filter(
                LiveTrade.resolved.is_(True),
                LiveTrade.resolved_at.isnot(None),
                LiveTrade.resolved_at >= today_start,
            ).all()
            today_pnl = sum(t.realized_pnl for t in today_trades if t.realized_pnl is not None)
            max_daily_loss = float(get_setting("max_daily_loss", "50.0") or 50.0)
            if max_daily_loss > 0 and today_pnl <= -max_daily_loss:
                logger.warning(
                    "Live trading paused: daily loss limit reached ($%.2f <= -$%.2f)",
                    today_pnl,
                    max_daily_loss,
                )
                return

            # Block live trades if no model is loaded — signals without a model
            # are unreliable and could place bad orders.
            try:
                get_model()
            except Exception:
                logger.error("Live trade skipped: no model loaded")
                return

            balance = get_balance()
            if balance is None:
                logger.error("Live trade skipped: cannot fetch Kalshi balance")
                return

            available = balance["balance_dollars"]
            if available < 1.0:
                logger.warning("Live trade skipped: insufficient balance $%.2f", available)
                return

            ticker = str(snapshot["market_ticker"])
            existing = LiveTrade.query.filter_by(ticker=ticker, resolved=False).first()
            if existing:
                logger.info(
                    "Live trade skipped: open %s position already exists on %s",
                    existing.side,
                    ticker,
                )
                return

            seconds_left = math.ceil(float(snapshot.get("seconds_to_close", 0) or 0))
            if seconds_left < MIN_SECONDS_FOR_AUTO_TRADE:
                logger.info(
                    "Live trade skipped: only %ss to close, minimum is %ss",
                    seconds_left,
                    MIN_SECONDS_FOR_AUTO_TRADE,
                )
                return

            live_size = float(get_setting("live_trade_size", "5.0") or 5.0)
            max_risk = available * 0.10

            if result.signal == "PAPER BUY YES":
                side = "yes"
                entry_price = float(result.p_market)
            else:
                side = "no"
                entry_price = 1.0 - float(result.p_market)

            if entry_price < float(MIN_ENTRY_PRICE):
                logger.warning(
                    "Live trade skipped: entry price %.3f below %.3f minimum (extreme leverage)",
                    entry_price,
                    float(MIN_ENTRY_PRICE),
                )
                return

            # --- Edge-based + upside-adjusted sizing (Kelly-lite) ---
            # Two factors scale position size:
            # 1. Model edge: how much the model disagrees with the market.
            #    Small edges → smaller positions; strong edges → boosted.
            # 2. Upside per contract: at 72¢ YES you win 28¢, at 28¢ NO you win 72¢.
            #    Flat sizing means 28¢ upside gets the same bet as 72¢ upside —
            #    so we scale by upside to equalize risk/reward across entry prices.
            # IMPORTANT: max_risk cap is applied AFTER multipliers so the final
            # trade size never exceeds 10% of available balance.
            model_prob = float(result.p_raw)
            market_prob = float(result.p_market)
            edge = abs(model_prob - market_prob)
            if edge < 0.05:
                logger.info(
                    "Live trade skipped: edge %.1f%% too small (model %.1f%% vs market %.1f%%) "
                    "— aggressive offset would eat the entire edge.",
                    edge * 100,
                    model_prob * 100,
                    market_prob * 100,
                )
                return
            if edge >= 0.15:
                edge_mult = 1.5
            elif edge >= 0.10:
                edge_mult = 1.0
            else:
                edge_mult = 0.5

            # Upside multiplier: normalize so 50¢ upside = 1.0x.
            # At 72¢ YES: upside 28¢ → 0.56x (small bet, small potential win)
            # At 55¢ YES: upside 45¢ → 0.90x (moderate)
            # At 28¢ NO:  upside 72¢ → 1.44x (large bet, large potential win)
            # Floor at 0.3x so we never bet tiny amounts on very expensive entries.
            # Cap at 1.0x so NO-side trades don't overshoot on mid-range entries.
            upside = (1.0 - entry_price) if side == "yes" else entry_price
            upside_mult = max(0.3, min(1.0, upside / 0.50))
            trade_size = live_size * edge_mult * upside_mult
            # Apply risk cap AFTER all multipliers so we never exceed 10% of balance.
            trade_size = min(trade_size, max_risk)
            logger.info(
                "Edge %.1f%% → %.1fx, upside %.0f¢ → %.2fx → $%.2f (cap $%.2f)",
                edge * 100,
                edge_mult,
                upside * 100,
                upside_mult,
                trade_size,
                max_risk,
            )

            contracts = int(trade_size / entry_price)
            if contracts < 1:
                logger.warning(
                    "Live trade skipped: trade size $%.2f too small for 1 contract at %.2f%%",
                    trade_size,
                    entry_price * 100,
                )
                return

            # --- Side-aware aggressive offset ---
            # YES books are liquid at 65-80¢; +2¢ crosses the spread.
            # NO books at 20-40¢ are very thin; need +5¢ to find liquidity.
            if side == "yes":
                raw_cents = round(result.p_market * 100)
                aggressive_offset = 2
                price_cents = max(1, min(99, raw_cents + aggressive_offset))
            else:
                raw_cents = round((1.0 - result.p_market) * 100)
                aggressive_offset = 5 if raw_cents <= 40 else 3
                price_cents = max(1, min(99, raw_cents + aggressive_offset))
            actual_cost = contracts * entry_price

            logger.info(
                "Placing live order: %s %s contracts on %s at %sc (+%dc offset, $%.2f risk, balance $%.2f)",
                side.upper(),
                contracts,
                ticker,
                price_cents,
                aggressive_offset,
                actual_cost,
                available,
            )

            order_result = place_order(
                ticker=ticker,
                side=side,
                count=contracts,
                price_cents=price_cents,
            )

            # --- Retry once at a worse price if the IOC didn't fill ---
            # The order book may have few contracts at our price; a second attempt
            # at a deeper price often finds liquidity.
            # Cap the retry price to avoid chasing into low-upside territory.
            max_entry_yes = float(get_setting("max_entry_price_yes", "0.80") or 0.80)
            if side == "yes":
                retry_cap_cents = max(1, int(max_entry_yes * 100))
                retry_offset = 3
            else:
                # For NO side, cap in NO-cents terms: (1 - min_YES_price) * 100
                # i.e. don't let NO price exceed max_entry_price_no in cents
                max_entry_no = float(get_setting("max_entry_price_no", "0.80") or 0.80)
                retry_cap_cents = max(1, int(max_entry_no * 100))
                retry_offset = 5 if raw_cents <= 40 else 3
            if order_result.get("unfilled"):
                retry_cents = min(price_cents + retry_offset, retry_cap_cents)
                if retry_cents > price_cents:
                    logger.info(
                        "Retrying unfilled order: %sc → %sc (+%dc retry, cap %sc)",
                        price_cents,
                        retry_cents,
                        retry_offset,
                        retry_cap_cents,
                    )
                    price_cents = retry_cents
                    # Recalculate entry_price and cost for the higher price
                    if side == "yes":
                        entry_price = price_cents / 100.0
                    else:
                        entry_price = 1.0 - (price_cents / 100.0)
                    actual_cost = contracts * entry_price
                    order_result = place_order(
                        ticker=ticker,
                        side=side,
                        count=contracts,
                        price_cents=price_cents,
                    )

            # Track fill rate for observability.
            try:
                attempts = int(get_setting("live_fill_attempts", "0") or 0)
                set_setting("live_fill_attempts", str(attempts + 1))
            except Exception:
                logger.debug("Failed to increment live_fill_attempts", exc_info=True)

            if order_result.get("success"):
                # Track successful fill for observability.
                try:
                    successes = int(get_setting("live_fill_successes", "0") or 0)
                    set_setting("live_fill_successes", str(successes + 1))
                except Exception:
                    logger.debug("Failed to increment live_fill_successes", exc_info=True)

                filled_contracts = int(order_result.get("fill_count") or contracts)
                fill_cost = order_result.get("fill_cost_dollars")
                avg_fill = order_result.get("average_fill_price")
                if avg_fill is not None:
                    entry_price = float(avg_fill)
                    price_cents = max(1, min(99, int(round(entry_price * 100))))
                if fill_cost is not None:
                    actual_cost = float(fill_cost)
                else:
                    actual_cost = filled_contracts * entry_price
                contracts = filled_contracts
                order_status = "placed"
                error_detail = None
            elif order_result.get("unfilled"):
                contracts = 0
                actual_cost = 0.0
                order_status = "unfilled"
                error_detail = order_result.get("error") or "No contracts filled"
            else:
                order_status = "failed"
                error_detail = order_result.get("error")
                if order_result.get("detail"):
                    error_detail = f"{error_detail}: {order_result.get('detail')}"

            signal_id = saved_signal.id if saved_signal is not None else None
            live_trade = LiveTrade(
                ticker=ticker,
                side=side.upper(),
                contracts=contracts,
                entry_price=entry_price,
                entry_price_cents=price_cents,
                cost_dollars=actual_cost,
                kalshi_order_id=order_result.get("order_id"),
                order_status=order_status,
                signal_id=signal_id,
                p_market_at_entry=result.p_market,
                p_raw_at_entry=result.p_raw,
                agreement_region=result.agreement_region,
                live_trade_size_setting=live_size,
                error_detail=error_detail,
            )
            db.session.add(live_trade)
            db.session.commit()

            if order_result.get("success"):
                logger.info(
                    "LIVE TRADE RECORDED: %s %s contracts on %s — order_id=%s",
                    side.upper(),
                    contracts,
                    ticker,
                    order_result.get("order_id"),
                )
            elif order_result.get("unfilled"):
                logger.warning(
                    "LIVE ORDER UNFILLED (recorded): %s on %s — order_id=%s",
                    error_detail,
                    ticker,
                    order_result.get("order_id"),
                )
            else:
                detail = order_result.get("detail", "")
                logger.error(
                    "LIVE ORDER FAILED (recorded for audit): %s — %s",
                    order_result.get("error"),
                    detail,
                )
        except Exception:
            logger.exception("Live trade execution error")


def poll_and_signal() -> None:
    """Scheduled polling job that computes and stores the latest signal."""
    global _latest_snapshot, _latest_signal, _consecutive_failures, _cooldown_until_ts, _cooldown_level
    start = time.time()
    try:
        if _app is None:
            logger.error("poll_and_signal called before scheduler app initialization.")
            return

        with _app.app_context():
            now_ts = time.time()
            if _cooldown_until_ts > now_ts:
                wait_left = int(max(1, _cooldown_until_ts - now_ts))
                logger.warning("Skipping poll during cooldown (%ss remaining).", wait_left)
                return

            if get_setting("scheduler_running", "false") == "false":
                logger.debug("Scheduler paused")
                return

            snapshot = get_live_snapshot()
            if snapshot is None:
                _consecutive_failures += 1
                logger.warning("No live snapshot available; skipping poll cycle.")
                if _consecutive_failures >= _MAX_FAILURES_BEFORE_COOLDOWN:
                    cooldown_secs = _COOLDOWN_SCHEDULE[min(_cooldown_level, len(_COOLDOWN_SCHEDULE) - 1)]
                    _cooldown_until_ts = time.time() + cooldown_secs
                    _cooldown_level += 1
                    logger.warning(
                        "%s consecutive failures. Cooling down for %ss (level %s).",
                        _consecutive_failures,
                        cooldown_secs,
                        _cooldown_level,
                    )
                    _consecutive_failures = 0
                return
            _consecutive_failures = 0
            _cooldown_level = 0
            logger.debug("Live snapshot received with keys: %s", sorted(snapshot.keys()))

            result = evaluate_live_signal(snapshot)
            if result is None:
                logger.warning("Live signal evaluation unavailable; model may not be loaded.")
                return
            logger.debug("Signal evaluation result: %s", signal_to_dict(result))
            logger.info(
                "Signal result: %s, region=%s, reversal_risk=%.3f",
                result.signal,
                result.agreement_region,
                float(snapshot.get("reversal_risk", 0.0) or 0.0),
            )
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

            live_setting_on = get_setting("live_trading_enabled", "false") == "true"
            live_keys_ok = kalshi_configured()
            live_enabled = live_setting_on and live_keys_ok
            actionable = result.signal in ("PAPER BUY YES", "PAPER BUY NO")

            # Live takes precedence: when live is on + keys configured, only place real orders.
            if live_enabled and actionable:
                logger.info("Live trading active — skipping paper auto-trade for this signal")
                _execute_live_trade(result, snapshot, saved_signal, _app)
            elif live_setting_on and not live_keys_ok and actionable:
                logger.warning(
                    "Live trading enabled but API keys not configured; falling back to paper auto-trade"
                )

            auto_trade_enabled = get_setting("auto_trade_enabled", "false") == "true"
            paper_trading_enabled = get_setting("paper_trading_enabled", "false") == "true"
            paper_auto_allowed = not (live_setting_on and live_keys_ok)
            if (
                paper_auto_allowed
                and auto_trade_enabled
                and paper_trading_enabled
                and actionable
            ):
                if _auto_trade_allowed_by_daily_loss():
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
                        seconds_left = math.ceil(float(snapshot.get("seconds_to_close", 0) or 0))
                        if seconds_left < MIN_SECONDS_FOR_AUTO_TRADE:
                            logger.info(
                                "Auto-trade skipped: only %ss to close, minimum is %ss",
                                seconds_left,
                                MIN_SECONDS_FOR_AUTO_TRADE,
                            )
                        else:
                            dynamic_sizing = get_setting("dynamic_sizing_enabled", "false") == "true"
                            dollar_amount = float(get_setting("paper_trade_size", "10.0"))
                            mode_str = (get_setting("signal_mode", "agreement") or "agreement").lower()
                            mispricing_gap = (
                                abs(float(result.p_raw) - float(result.p_market))
                                if mode_str == "mispricing"
                                else 0.0
                            )

                            entry_price = float(result.p_market or 0.0) if side == "YES" else (1.0 - float(result.p_market or 0.0))
                            contracts = (dollar_amount / entry_price) if entry_price > 0 else 0.0
                            if contracts > 0:
                                open_position = PaperTrade.query.filter(
                                    PaperTrade.ticker == str(snapshot["market_ticker"]),
                                    PaperTrade.resolved.is_(False),
                                ).first()
                                if open_position:
                                    logger.info(
                                        "Auto-trade skipped: open %s position already exists on %s",
                                        open_position.side,
                                        snapshot["market_ticker"],
                                    )
                                else:
                                    price_ctx = _price_context_for_snapshot()
                                    snapshot_data = {
                                        "market_title": snapshot.get("market_title", "") or "",
                                        "seconds_to_close": math.ceil(float(snapshot.get("seconds_to_close", 0) or 0)),
                                        "entry_bucket": int(snapshot.get("entry_bucket", 0) or 0),
                                        "p_market": float(snapshot.get("p_market", 0) or 0),
                                        "p_raw": float(snapshot.get("p_raw", 0) or 0),
                                        "signal_mode": get_setting("signal_mode", "agreement") or "agreement",
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
                                        volatility_override=("volatility override" in str(result.reason).lower()),
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
    finally:
        elapsed = time.time() - start
        logger.info("poll_and_signal completed in %.2fs", elapsed)
        if elapsed > 5.0:
            logger.warning("Slow poll: %.2fs — API may be slow", elapsed)


def resolve_job() -> None:
    """Scheduled resolution job to settle recently closed markets."""
    try:
        if _app is None:
            logger.error("resolve_job called before scheduler app initialization.")
            return
        with _app.app_context():
            count = resolve_pending_markets()
            live_resolved = resolve_live_trades()
            logger.info("Resolution check complete, %s markets resolved", count)
            if live_resolved > 0:
                logger.info("Resolved %s live trade(s)", live_resolved)
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

    # Fixed 30s poll: keeps Kalshi retry headroom (~3s) under max_instances=1 without overlap.
    poll_seconds = 30

    scheduler_instance = BackgroundScheduler(timezone="UTC")
    scheduler_instance.add_job(
        poll_and_signal,
        trigger="interval",
        seconds=poll_seconds,
        id="poll_signal",
        replace_existing=True,
        misfire_grace_time=10,
        coalesce=True,
        max_instances=1,
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
    logger.info("Scheduler started, polling every %ss", poll_seconds)

    def _warmup_cache() -> None:
        time.sleep(2)
        logger.info("Warming up Kalshi API cache...")
        try:
            if _app is None:
                return
            with _app.app_context():
                poll_and_signal()
            logger.info("Cache warmed")
        except Exception:
            logger.exception("Cache warmup failed")

    threading.Thread(target=_warmup_cache, daemon=True, name="kalshi-cache-warmup").start()

    _SCHEDULER_INSTANCE = scheduler_instance
    return scheduler_instance
