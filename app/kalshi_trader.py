"""Live Kalshi order placement and portfolio queries."""

from __future__ import annotations

import logging

import requests

from app.kalshi_auth import TRADING_BASE_URL, get_kalshi_headers, is_configured

logger = logging.getLogger(__name__)


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
    body = {
        "ticker": ticker,
        "action": "buy",
        "side": side,
        "type": "limit",
        "count": count,
        "limit_price": price_cents,
        "time_in_force": "immediate_or_cancel",
    }
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
            order_id = data.get("order", {}).get("order_id", "unknown")
            logger.info(
                "LIVE ORDER PLACED: %s %s contracts on %s at %sc — order_id=%s",
                side.upper(),
                count,
                ticker,
                price_cents,
                order_id,
            )
            return {"success": True, "order": data, "order_id": order_id}
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
