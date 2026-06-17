"""Live Kalshi order placement and portfolio queries."""

from __future__ import annotations

import logging

import requests

from app.kalshi_auth import TRADING_BASE_URL, get_kalshi_headers, is_configured

logger = logging.getLogger(__name__)


def _parse_fp_count(value) -> float:
    if value is None:
        return 0.0
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return 0.0


def _parse_fp_dollars(value) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _parse_order_fill(order_payload: dict) -> dict:
    """Extract fill count and cost from a Kalshi order response."""
    if not isinstance(order_payload, dict):
        order_payload = {}
    fill_count = _parse_fp_count(
        order_payload.get("fill_count_fp") or order_payload.get("fill_count")
    )
    fill_cost_dollars = _parse_fp_dollars(order_payload.get("taker_fill_cost_dollars"))
    if fill_cost_dollars is None and order_payload.get("taker_fill_cost") is not None:
        try:
            fill_cost_dollars = float(order_payload["taker_fill_cost"]) / 100.0
        except (TypeError, ValueError):
            fill_cost_dollars = None
    return {
        "fill_count": fill_count,
        "fill_cost_dollars": fill_cost_dollars,
        "status": order_payload.get("status"),
    }


def get_balance() -> dict | None:
    if not is_configured():
        return None
    path = "/portfolio/balance"
    headers = get_kalshi_headers("GET", path)
    if not headers:
        return None
    try:
        response = requests.get(TRADING_BASE_URL + path, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            balance_cents = data.get("balance", 0)
            return {
                "balance_cents": balance_cents,
                "balance_dollars": round(balance_cents / 100, 2),
                "raw": data,
            }
        logger.error("Balance check failed: %s %s", response.status_code, response.text[:200])
        return None
    except Exception as exc:
        logger.error("Balance check error: %s", exc)
        return None


def place_order(
    ticker: str,
    side: str,
    count: int,
    price_cents: int,
) -> dict:
    """Place a limit IOC buy order on Kalshi."""
    if not is_configured():
        return {"error": "API keys not configured"}
    if count < 1:
        return {"error": "Minimum 1 contract"}
    if not (1 <= price_cents <= 99):
        return {"error": f"Invalid price {price_cents}c — must be 1-99"}

    path = "/portfolio/orders"
    normalized_side = str(side).lower()
    body: dict = {
        "ticker": ticker,
        "action": "buy",
        "side": normalized_side,
        "type": "limit",
        "count": int(count),
        "time_in_force": "immediate_or_cancel",
    }
    if normalized_side == "yes":
        body["yes_price"] = int(price_cents)
    elif normalized_side == "no":
        body["no_price"] = int(price_cents)
    else:
        return {"error": f"Invalid side {side!r} — must be yes or no"}
    headers = get_kalshi_headers("POST", path)
    if not headers:
        return {"error": "Failed to generate auth headers"}
    try:
        response = requests.post(
            TRADING_BASE_URL + path,
            headers=headers,
            json=body,
            timeout=15,
        )
        if response.status_code in (200, 201):
            data = response.json()
            order_payload = data.get("order") if isinstance(data.get("order"), dict) else data
            order_id = (
                order_payload.get("order_id")
                or data.get("order_id")
                or "unknown"
            )
            fill = _parse_order_fill(order_payload)
            fill_count = int(fill["fill_count"]) if fill["fill_count"] >= 1 else 0
            if fill_count < 1:
                logger.warning(
                    "LIVE ORDER UNFILLED: %s %s contracts on %s at %sc — order_id=%s status=%s",
                    side.upper(),
                    count,
                    ticker,
                    price_cents,
                    order_id,
                    fill.get("status"),
                )
                return {
                    "success": False,
                    "unfilled": True,
                    "order_id": order_id,
                    "fill_count": 0,
                    "error": "No contracts filled (IOC order)",
                    "order": data,
                }
            logger.info(
                "LIVE ORDER PLACED: %s %s filled on %s at %sc — order_id=%s",
                side.upper(),
                fill_count,
                ticker,
                price_cents,
                order_id,
            )
            return {
                "success": True,
                "order": data,
                "order_id": order_id,
                "fill_count": fill_count,
                "fill_cost_dollars": fill["fill_cost_dollars"],
            }
        logger.error(
            "LIVE ORDER FAILED: %s — %s (ticker=%s, side=%s, count=%s, price=%sc)",
            response.status_code,
            response.text[:300],
            ticker,
            side,
            count,
            price_cents,
        )
        return {
            "error": f"Order failed: {response.status_code}",
            "detail": response.text[:300],
        }
    except Exception as exc:
        logger.exception("Order placement exception: %s", exc)
        return {"error": str(exc)}


def get_open_positions() -> list:
    """Fetch current open positions from Kalshi."""
    if not is_configured():
        return []
    path = "/portfolio/positions"
    headers = get_kalshi_headers("GET", path)
    if not headers:
        return []
    try:
        response = requests.get(TRADING_BASE_URL + path, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.json().get("market_positions", [])
        return []
    except Exception:
        return []


def get_settlement_for_ticker(ticker: str) -> dict | None:
    """Fetch Kalshi settlement for a market (source of truth for PnL)."""
    if not is_configured():
        return None
    path = "/portfolio/settlements"
    headers = get_kalshi_headers("GET", path)
    if not headers:
        return None
    try:
        response = requests.get(
            TRADING_BASE_URL + path,
            headers=headers,
            params={"ticker": ticker, "limit": 1},
            timeout=10,
        )
        if response.status_code != 200:
            logger.warning(
                "Settlement fetch failed for %s: %s %s",
                ticker,
                response.status_code,
                response.text[:200],
            )
            return None
        settlements = response.json().get("settlements") or []
        if not settlements:
            return None
        row = settlements[0]
        return row if isinstance(row, dict) else None
    except Exception as exc:
        logger.error("Settlement fetch error for %s: %s", ticker, exc)
        return None
