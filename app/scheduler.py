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
_COOLDOWN_SCHEDULE = [30, 60, 90, 120]


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


def _cancel_stale_resting_order(ticker: str, result, app=None) -> None:
    """Cancel a resting GTC order whose thesis the current signal no longer supports.

    A GTC bid is placed on a mispricing signal and rests until filled or expiry.
    If the market then moves so the live model no longer backs that side (the
    signal collapses to NO SIGNAL or flips to the other side), a still-resting bid
    would only fill on a move against us — adverse selection. We cancel it instead
    of waiting for expiry. Runs every poll for the active market regardless of
    whether the current signal is actionable. Filled/partially-filled orders are
    left untouched for the normal reconciliation path.
    """
    from app.models import LiveTrade, db
    from app.kalshi_trader import cancel_order, get_order_status

    existing = LiveTrade.query.filter(
        LiveTrade.ticker == ticker,
        LiveTrade.resolved.is_(False),
        LiveTrade.order_status == "resting",
        LiveTrade.kalshi_order_id.isnot(None),
    ).first()
    if existing is None:
        return

    supports = (
        (result.signal == "PAPER BUY YES" and str(existing.side).upper() == "YES")
        or (result.signal == "PAPER BUY NO" and str(existing.side).upper() == "NO")
    )
    if supports:
        return  # current signal still backs this side — keep waiting for the fill

    status = get_order_status(existing.kalshi_order_id)
    if status is None:
        return  # can't verify status — don't cancel blindly (it may have filled)
    if int(status.get("fill_count", 0) or 0) >= 1:
        return  # (partially) filled — leave for the normal reconciliation path

    cancel_order(existing.kalshi_order_id)
    # Mark "unfilled" (not "cancelled") so the resting-aware position check treats
    # it as no-position and allows a fresh evaluation, rather than a stale block.
    existing.order_status = "unfilled"
    existing.contracts = 0
    existing.error_detail = f"cancelled stale resting order (signal now {result.signal})"
    set_setting(f"live_resting_order_{ticker}", "")
    db.session.commit()
    logger.info(
        "Cancelled stale resting %s order %s on %s — current signal %s no longer supports it",
        existing.side,
        existing.kalshi_order_id,
        ticker,
        result.signal,
    )


def _aggressive_entry_price(side, p_market, quote, max_entry_yes, max_entry_no, buffer_cents=1):
    """Price a marketable order that CROSSES the live orderbook ask for an
    immediate fill, instead of resting a passive limit at the stale candle mid.

    Immediate fills raise the fill rate (~65% resting -> ~all) and remove the
    adverse selection of passive limits (which only fill when the market moves
    against us — the gap between paper 78% WR and live 59% WR). Backtest: crossing
    stays +EV up to ~11c/contract; observed real crossing cost is ~1-3c.

    Returns (entry_price, price_cents), or None when the live ask exceeds the
    max-entry sanity cap (never chase past the cap). Falls back to the candle mid
    when no live quote is available.
    """
    if side == "yes":
        mid = float(p_market)
        ask = quote.get("yes_ask") if quote else None
        cap = float(max_entry_yes)
    else:
        mid = 1.0 - float(p_market)
        ask = quote.get("no_ask") if quote else None
        cap = float(max_entry_no)
    base = float(ask) if ask is not None else mid
    if base > cap:
        return None
    price_cents = max(1, min(99, int(round(base * 100)) + int(buffer_cents)))
    return base, price_cents


def _apply_contract_cap(contracts: int, live_max_contracts) -> int:
    """Clamp contract count to live_max_contracts when it is a positive int.

    Empty string, None, non-numeric, or <=0 all mean "no cap". Used for the
    interim tiny-fixed-size posture during model retraining.
    """
    try:
        cap = int(str(live_max_contracts).strip())
    except (TypeError, ValueError):
        return contracts
    if cap <= 0:
        return contracts
    return min(contracts, cap)


def _execute_live_trade(result, snapshot, saved_signal, app) -> None:
    """Place a real Kalshi order mirroring the signal. All safety checks run first."""
    from datetime import datetime, timezone

    from app.kalshi_trader import get_balance, get_order_status, place_order
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

            # --- Resting-aware position check ---
            # Before placing a new order, check if we already have a trade on
            # this ticker.  For "placed" (filled) trades we skip — we have a
            # real position.  For "resting" (GTC in book) trades we query
            # Kalshi to see if they've filled since we last checked.
            # Note: We include "cancelled" and "expired" so we can clean them up.
            existing = LiveTrade.query.filter(
                LiveTrade.ticker == ticker,
                LiveTrade.resolved.is_(False),
                LiveTrade.order_status.notin_(["unfilled", "failed"]),
            ).first()

            if existing:
                if existing.order_status == "resting" and existing.kalshi_order_id:
                    # GTC order is resting — check if it has filled.
                    status = get_order_status(existing.kalshi_order_id)
                    if status is None:
                        # Can't reach Kalshi — assume still resting, don't
                        # place a duplicate.
                        logger.info(
                            "Live trade skipped: resting order %s on %s, "
                            "can't verify fill status — waiting",
                            existing.kalshi_order_id,
                            ticker,
                        )
                        return

                    kalshi_status = status.get("status", "unknown")
                    fill_count = status.get("fill_count", 0)

                    if fill_count >= 1:
                        # Order has (partially) filled while resting!
                        # Update the existing LiveTrade row.
                        existing.order_status = "placed"
                        existing.contracts = fill_count
                        avg_price = status.get("average_fill_price")
                        if avg_price is not None:
                            existing.entry_price = float(avg_price)
                            existing.entry_price_cents = max(
                                1, min(99, int(round(float(avg_price) * 100)))
                            )
                        fill_cost = status.get("fill_cost_dollars")
                        if fill_cost is not None:
                            existing.cost_dollars = float(fill_cost)
                        else:
                            existing.cost_dollars = fill_count * existing.entry_price
                        existing.error_detail = None
                        db.session.commit()
                        # Clear the resting-order tracker since it's now filled.
                        set_setting(f"live_resting_order_{ticker}", "")
                        logger.info(
                            "Resting GTC order %s FILLED: %s %s contracts on %s "
                            "(detected during position check)",
                            existing.kalshi_order_id,
                            existing.side,
                            fill_count,
                            ticker,
                        )
                        # We now have a real position — don't place another.
                        return

                    if kalshi_status in ("cancelled", "expired", "rejected"):
                        # Order was killed externally (or by the resolver for a
                        # closed market).  Mark it and allow a fresh order.
                        logger.info(
                            "Resting order %s on %s is %s — allowing new order",
                            existing.kalshi_order_id,
                            ticker,
                            kalshi_status,
                        )
                        existing.order_status = "cancelled"
                        set_setting(f"live_resting_order_{ticker}", "")
                        db.session.commit()
                        # Fall through to place a new order below.

                    else:
                        # Still resting / open — don't place a duplicate.
                        logger.info(
                            "Live trade skipped: GTC order %s still resting on %s "
                            "(status=%s, 0 fills) — waiting for fill",
                            existing.kalshi_order_id,
                            ticker,
                            kalshi_status,
                        )
                        return

                else:
                    # "placed" (filled) trade — real open position exists.
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

            side = "yes" if result.signal == "PAPER BUY YES" else "no"

            # Cross the LIVE orderbook ask for an immediate fill, rather than
            # resting a passive limit at the stale candle mid (which filled ~65%
            # and adversely selected). Sanity-capped by max-entry so we never chase.
            quote = get_market_prices(ticker)
            max_entry_yes = float(get_setting("max_entry_price_yes", "0.65") or 0.65)
            max_entry_no = float(get_setting("max_entry_price_no", "0.80") or 0.80)
            priced = _aggressive_entry_price(side, result.p_market, quote, max_entry_yes, max_entry_no)
            if priced is None:
                logger.info(
                    "Live trade skipped: live %s ask exceeds max-entry cap (quote=%s)",
                    side.upper(),
                    quote,
                )
                return
            entry_price, price_cents = priced

            if entry_price < float(MIN_ENTRY_PRICE):
                logger.warning(
                    "Live trade skipped: entry price %.3f below %.3f minimum (extreme leverage)",
                    entry_price,
                    float(MIN_ENTRY_PRICE),
                )
                return

            # --- Edge-based + upside-adjusted sizing (kelly_lite_v2) ---
            # Evidence from backtest_v2.py (2026-07-07):
            #   - kelly_lite_v2 reduces max drawdown ($5.20 vs $7.89 baseline)
            #     while maintaining comparable Sharpe (0.158 vs 0.128 baseline)
            #   - Key change: upside_mult capped at 0.8 (not 1.0) to prevent
            #     overexposure on cheap NO entries; hard $4 cap prevents blow-up
            #     on large-size trades that empirically all lost.
            model_prob  = float(result.p_raw)
            market_prob = float(result.p_market)
            edge = abs(model_prob - market_prob)
            is_mispricing = "mispricing" in (result.agreement_region or "").lower() or edge >= 0.10

            # Skip mispricing signals with very small edges — the aggressive
            # spread offset would eat the entire margin.
            if is_mispricing and edge < 0.05:
                logger.info(
                    "Live trade skipped: mispricing edge %.1f%% too small "
                    "(model %.1f%% vs market %.1f%%)",
                    edge * 100,
                    model_prob * 100,
                    market_prob * 100,
                )
                return

            # Edge multiplier: higher confidence → slightly larger bet.
            if edge >= 0.20:
                edge_mult = 1.5
            elif edge >= 0.15:
                edge_mult = 1.2
            elif edge >= 0.10:
                edge_mult = 1.0
            else:
                edge_mult = 0.7 if not is_mispricing else 0.4

            # Upside multiplier: normalize so 50¢ upside = 1.0x.
            # Capped at 0.8 (vs 1.0 previously) to control NO-side exposure.
            # Floor at 0.3x so expensive-entry trades stay small.
            upside = (1.0 - entry_price) if side == "yes" else entry_price
            upside_mult = max(0.3, min(0.8, upside / 0.50))

            # Hard $4 cap: backtest showed all trades > $3.50 were losses.
            trade_size = min(live_size * edge_mult * upside_mult, 4.0, max_risk)
            logger.info(
                "Edge %.1f%% → %.1fx, upside %.0f¢ → %.2fx → $%.2f (cap $%.2f/$%.2f)",
                edge * 100,
                edge_mult,
                upside * 100,
                upside_mult,
                trade_size,
                max_risk,
                4.0,
            )

            contracts = int(trade_size / entry_price)
            if contracts < 1:
                logger.warning(
                    "Live trade skipped: trade size $%.2f too small for 1 contract at %.2f%%",
                    trade_size,
                    entry_price * 100,
                )
                return

            live_max_contracts = get_setting("live_max_contracts", "")
            capped = _apply_contract_cap(contracts, live_max_contracts)
            if capped != contracts:
                logger.info(
                    "Contract cap: %s -> %s (live_max_contracts=%s)",
                    contracts, capped, live_max_contracts,
                )
                contracts = capped

            # price_cents (crossing the live ask) was computed above by
            # _aggressive_entry_price; entry_price is the expected fill price.
            actual_cost = contracts * entry_price

            # Calculate expiration time (60s before market close)
            expiration_ts = int(snapshot["close_ts"]) - 60
            now_ts = int(time.time())
            time_until_expiration = expiration_ts - now_ts
            
            # Safety check: don't place orders that expire in <30s
            # (orders already expire 60s before market close, so this ensures
            # at least 30s for the order to fill after being placed)
            if time_until_expiration < 30:
                logger.warning(
                    "Live trade skipped: order would expire in %ss (min 30s required)",
                    time_until_expiration,
                )
                return
            
            logger.info(
                "Placing live GTC order (crossing ask): %s %s contracts on %s at %sc "
                "($%.2f risk, balance $%.2f, expires in %ss)",
                side.upper(),
                contracts,
                ticker,
                price_cents,
                actual_cost,
                available,
                time_until_expiration,
            )

            order_result = place_order(
                ticker=ticker,
                side=side,
                count=contracts,
                price_cents=price_cents,
                gtc=True,
                expiration_ts=expiration_ts,
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

                # If GTC order is resting (not yet filled), save its ID so we
                # can cancel it before placing the next order on the same ticker.
                if order_result.get("resting") and not filled_contracts:
                    order_id = order_result.get("order_id", "unknown")
                    set_setting(f"live_resting_order_{ticker}", order_id)
                    logger.info("GTC order %s resting on %s — will check fill status on next poll", order_id, ticker)
                    contracts = 0
                    actual_cost = 0.0
                    order_status = "resting"
                    error_detail = None
                else:
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

            # Adverse-selection hygiene: cancel any resting GTC order on this ticker
            # that the current signal no longer supports, before (maybe) placing a
            # new one. Runs regardless of whether the current signal is actionable.
            if live_enabled:
                _cancel_stale_resting_order(ticker=str(snapshot["market_ticker"]), result=result)

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


def cleanup_resting_orders_job() -> None:
    """Sync status of all resting GTC orders from Kalshi and clean up expired/cancelled ones."""
    try:
        if _app is None:
            logger.error("cleanup_resting_orders_job called before scheduler app initialization.")
            return
        
        from app.kalshi_trader import get_order_status
        from app.models import LiveTrade, db
        
        with _app.app_context():
            # Find all resting orders that haven't been resolved yet
            resting_orders = LiveTrade.query.filter(
                LiveTrade.order_status == "resting",
                LiveTrade.resolved.is_(False),
            ).all()
            
            if not resting_orders:
                logger.debug("No resting orders to clean up")
                return
            
            logger.info("Checking status of %s resting order(s)", len(resting_orders))
            cleaned = 0
            filled = 0
            
            for trade in resting_orders:
                if not trade.kalshi_order_id:
                    logger.warning("Resting trade %s has no order_id, marking as failed", trade.id)
                    trade.order_status = "failed"
                    trade.error_detail = "No order ID recorded"
                    cleaned += 1
                    continue
                
                status = get_order_status(trade.kalshi_order_id)
                if status is None:
                    logger.debug("Could not fetch status for order %s, skipping", trade.kalshi_order_id)
                    continue
                
                kalshi_status = status.get("status", "unknown")
                fill_count = status.get("fill_count", 0)
                
                if fill_count >= 1:
                    # Order filled while resting!
                    trade.order_status = "placed"
                    trade.contracts = fill_count
                    avg_price = status.get("average_fill_price")
                    if avg_price is not None:
                        trade.entry_price = float(avg_price)
                        trade.entry_price_cents = max(1, min(99, int(round(float(avg_price) * 100))))
                    fill_cost = status.get("fill_cost_dollars")
                    if fill_cost is not None:
                        trade.cost_dollars = float(fill_cost)
                    else:
                        trade.cost_dollars = fill_count * trade.entry_price
                    trade.error_detail = None
                    # Clear the resting order tracker
                    set_setting(f"live_resting_order_{trade.ticker}", "")
                    logger.info(
                        "Resting order %s FILLED during cleanup: %s %s contracts on %s",
                        trade.kalshi_order_id,
                        trade.side,
                        fill_count,
                        trade.ticker,
                    )
                    filled += 1
                elif kalshi_status in ("cancelled", "expired", "rejected"):
                    # Order was killed - clean it up
                    trade.order_status = kalshi_status
                    if kalshi_status == "expired":
                        trade.error_detail = "Order expired before fill"
                    elif kalshi_status == "cancelled":
                        trade.error_detail = "Order cancelled externally"
                    else:
                        trade.error_detail = f"Order {kalshi_status}"
                    set_setting(f"live_resting_order_{trade.ticker}", "")
                    logger.info(
                        "Resting order %s is %s, cleaned up",
                        trade.kalshi_order_id,
                        kalshi_status,
                    )
                    cleaned += 1
            
            if cleaned > 0 or filled > 0:
                db.session.commit()
                logger.info(
                    "Cleanup complete: %s order(s) filled, %s order(s) cleaned up",
                    filled,
                    cleaned,
                )
    except Exception:
        logger.exception("cleanup_resting_orders_job failed")


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

    # 45s poll: reduced from 30s to ease Kalshi rate-limit pressure.
    # With 15-min markets this gives ~20 poll opportunities per market — plenty for GTC orders.
    poll_seconds = 45

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
    scheduler_instance.add_job(
        cleanup_resting_orders_job,
        trigger="interval",
        minutes=5,
        id="cleanup_resting_orders",
        replace_existing=True,
        misfire_grace_time=30,
    )
    scheduler_instance.start()
    logger.info("Scheduler started, polling every %ss, cleanup every 5min", poll_seconds)

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
