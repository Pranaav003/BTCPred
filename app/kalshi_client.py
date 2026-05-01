"""Defensive Kalshi REST client for market/trade data retrieval."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from threading import Lock
from typing import Any

import pandas as pd
import requests

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
SERIES = "KXBTC15M"
REQUEST_TIMEOUT = 20
PREFERRED_STRIKE = "-30"

logger = logging.getLogger(__name__)
_btc_price_cache: dict[str, float | None] = {"price": None, "ts": 0}
_btc_429_until = 0.0
BTC_PRICE_CACHE_TTL = 60.0

_cache_lock = Lock()
_market_cache: dict[str, dict[str, Any]] = {}
_candle_cache: dict[str, dict[str, Any]] = {}
_trade_cache: dict[str, dict[str, Any]] = {}
_request_lock = Lock()
_last_request_time = 0.0
_min_request_interval = 1.0

MARKET_CACHE_TTL = 30.0
CANDLE_CACHE_TTL = 12.0
TRADE_CACHE_TTL = 10.0
_MAX_CANDLE_CACHE_ENTRIES = 10
_MAX_TRADE_CACHE_ENTRIES = 20
# NOTE: The following endpoints require authentication (API key):
# - GET /markets/{ticker}/orderbook
# - POST /portfolio/orders (order placement)
# - GET /portfolio/balance
# These are not implemented until KALSHI_API_KEY is configured.
# See .env.example for KALSHI_API_KEY and KALSHI_API_SECRET.


def _get(url: str, params: dict[str, Any] | None = None, max_retries: int = 3) -> dict | None:
    """GET JSON helper with throttling/retries; returns None on failure."""
    global _last_request_time
    for attempt in range(max_retries):
        try:
            # Global request pacing to reduce 429s under concurrent polling.
            with _request_lock:
                now = time.time()
                elapsed = now - _last_request_time
                if elapsed < _min_request_interval:
                    time.sleep(_min_request_interval - elapsed)
                _last_request_time = time.time()

            response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            if response.status_code == 429:
                wait = (2**attempt) * 5
                logger.warning(
                    "Rate limited by Kalshi (attempt %s/%s). Waiting %ss before retry.",
                    attempt + 1,
                    max_retries,
                    wait,
                )
                time.sleep(wait)
                continue

            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                logger.error("Kalshi API returned non-dict JSON for %s", url)
                return None
            return payload
        except requests.exceptions.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            if status_code == 429 and attempt < max_retries - 1:
                wait = (2**attempt) * 5
                logger.warning("Kalshi HTTP 429 retry in %ss for %s", wait, url)
                time.sleep(wait)
                continue
            logger.error("Kalshi GET failed url=%s params=%s error=%s", url, params, exc)
            return None
        except Exception as exc:
            logger.error("Kalshi GET failed url=%s params=%s error=%s", url, params, exc)
            return None

    logger.error("Kalshi GET failed after %s retries: url=%s params=%s", max_retries, url, params)
    return None


def _to_unix(value: Any) -> int | None:
    """Convert int/float/ISO datetime input to unix timestamp seconds."""
    if value is None:
        return None

    try:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            raw = value.strip()
            if raw == "":
                return None
            if raw.isdigit():
                return int(raw)
            if raw.endswith("Z"):
                raw = raw[:-1] + "+00:00"
            dt = datetime.fromisoformat(raw)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp())
    except Exception:
        return None

    return None


def _normalize_price(value: Any) -> float | None:
    """Normalize price input to probability in [0, 1] if possible."""
    candidate: Any = value

    if isinstance(value, dict):
        for key in ("close", "close_dollars", "yes_bid", "yes_ask"):
            if key in value and value.get(key) is not None:
                candidate = value.get(key)
                break
        else:
            return None

    if isinstance(candidate, dict):
        for key in ("close", "close_dollars"):
            if key in candidate and candidate.get(key) is not None:
                candidate = candidate.get(key)
                break
        else:
            return None

    try:
        if candidate is None or isinstance(candidate, bool):
            return None
        if isinstance(candidate, str):
            candidate = candidate.strip()
            if candidate == "":
                return None
        price = float(candidate)
        if price > 1.0:
            price = price / 100.0
        return price
    except Exception:
        return None


def _fetch_active_market(preferred_suffix: str = PREFERRED_STRIKE) -> dict | None:
    """Return nearest KXBTC15M market closing within the next 900 seconds."""
    url = f"{BASE_URL}/markets"
    payload = _get(url, {"series_ticker": SERIES, "limit": 200})
    if payload is None:
        return None

    markets = payload.get("markets")
    if not isinstance(markets, list):
        logger.error("Unexpected markets payload shape.")
        return None

    now_ts = int(datetime.now(timezone.utc).timestamp())
    max_ts = now_ts + 900
    candidates: list[dict[str, Any]] = []

    for market in markets:
        if not isinstance(market, dict):
            continue
        ticker = market.get("ticker")
        if not isinstance(ticker, str) or not ticker.startswith(SERIES):
            continue
        close_iso = market.get("close_time")
        close_ts = _to_unix(close_iso)
        if close_ts is None:
            continue
        if now_ts <= close_ts <= max_ts:
            candidates.append(
                {
                    "ticker": ticker,
                    "title": market.get("title"),
                    "close_time_iso": close_iso,
                    "close_ts": close_ts,
                    "series_ticker": market.get("series_ticker", SERIES),
                }
            )

    if not candidates:
        return None

    candidates.sort(key=lambda row: row["close_ts"])
    preferred = [m for m in candidates if str(m.get("ticker", "")).endswith(preferred_suffix)]
    selected = preferred if preferred else candidates
    return selected[0]


def get_active_market(preferred_suffix: str = PREFERRED_STRIKE) -> dict | None:
    """Return nearest KXBTC15M market; responses cached briefly to reduce API latency."""
    suffix_key = str(preferred_suffix)
    with _cache_lock:
        now = time.time()
        cached = _market_cache.get(suffix_key)
        if cached and cached.get("data") is not None and now - float(cached.get("ts") or 0) < MARKET_CACHE_TTL:
            return cached["data"]

    result = _fetch_active_market(preferred_suffix)

    with _cache_lock:
        _market_cache[suffix_key] = {"data": result, "ts": time.time()}
        if len(_market_cache) > 8:
            oldest = min(_market_cache, key=lambda k: float(_market_cache[k].get("ts") or 0))
            del _market_cache[oldest]
    return result


def _fetch_candles(ticker: str, close_ts: int) -> pd.DataFrame:
    """Fetch minute candles for last 900 seconds before close."""
    columns = ["ts", "close", "high", "low"]
    empty = pd.DataFrame(columns=columns)

    try:
        start_ts = int(close_ts) - 900
        end_ts = int(close_ts)
    except Exception:
        logger.error("Invalid close_ts for get_candles: %s", close_ts)
        return empty

    url = f"{BASE_URL}/series/{SERIES}/markets/{ticker}/candlesticks"
    payload = _get(url, {"start_ts": start_ts, "end_ts": end_ts, "period_interval": 1})
    if payload is None:
        return empty

    candles = payload.get("candlesticks")
    if not isinstance(candles, list):
        logger.error("Unexpected candlesticks payload shape.")
        return empty

    rows: list[dict[str, Any]] = []
    for candle in candles:
        if not isinstance(candle, dict):
            continue

        ts = _to_unix(candle.get("end_period_ts"))
        price = candle.get("price") if isinstance(candle.get("price"), dict) else {}

        close = _normalize_price(price.get("close"))
        if close is None:
            close = _normalize_price(price.get("close_dollars"))
        if close is None:
            yes_bid = _normalize_price(price.get("yes_bid"))
            yes_ask = _normalize_price(price.get("yes_ask"))
            if yes_bid is not None and yes_ask is not None:
                close = (yes_bid + yes_ask) / 2.0

        high = _normalize_price(price.get("high"))
        if high is None:
            high = _normalize_price(price.get("high_dollars"))
        if high is None:
            high = close

        low = _normalize_price(price.get("low"))
        if low is None:
            low = _normalize_price(price.get("low_dollars"))
        if low is None:
            low = close

        rows.append({"ts": ts, "close": close, "high": high, "low": low})

    if not rows:
        return empty

    df = pd.DataFrame(rows, columns=columns)
    df = df.dropna(subset=["ts", "close"]).sort_values("ts").reset_index(drop=True)
    return df if not df.empty else empty


def get_candles(ticker: str, close_ts: int) -> pd.DataFrame:
    """Fetch minute candles; results cached briefly (same inputs → same snapshot logic)."""
    cache_key = f"{ticker}_{int(close_ts)}"
    with _cache_lock:
        now = time.time()
        cached = _candle_cache.get(cache_key)
        if cached and now - float(cached.get("ts") or 0) < CANDLE_CACHE_TTL:
            df = cached.get("data")
            if isinstance(df, pd.DataFrame):
                return df.copy()

    result = _fetch_candles(ticker, close_ts)

    with _cache_lock:
        _candle_cache[cache_key] = {"data": result, "ts": time.time()}
        if len(_candle_cache) > _MAX_CANDLE_CACHE_ENTRIES:
            oldest = min(_candle_cache, key=lambda k: float(_candle_cache[k].get("ts") or 0))
            del _candle_cache[oldest]
    return result.copy() if isinstance(result, pd.DataFrame) else result


def _fetch_trades(ticker: str, start_ts: int, end_ts: int) -> pd.DataFrame:
    """Fetch and paginate trades between timestamps."""
    columns = ["ts", "price", "qty"]
    empty = pd.DataFrame(columns=columns)
    url = f"{BASE_URL}/markets/trades"

    cursor: str | None = None
    all_rows: list[dict[str, Any]] = []

    while True:
        params: dict[str, Any] = {
            "ticker": ticker,
            "min_ts": start_ts,
            "max_ts": end_ts,
            "limit": 1000,
        }
        if cursor:
            params["cursor"] = cursor

        payload = _get(url, params)
        if payload is None:
            return empty

        trades = payload.get("trades")
        if not isinstance(trades, list):
            logger.error("Unexpected trades payload shape.")
            return empty

        for trade in trades:
            if not isinstance(trade, dict):
                continue
            ts = _to_unix(trade.get("created_time"))
            if ts is None:
                ts = _to_unix(trade.get("created_ts"))
            if ts is None:
                ts = _to_unix(trade.get("ts"))

            price = _normalize_price(trade.get("yes_price"))
            if price is None:
                price = _normalize_price(trade.get("yes_price_dollars"))
            if price is None:
                price = _normalize_price(trade.get("price"))

            qty_raw = (
                trade.get("count")
                if trade.get("count") is not None
                else trade.get("quantity")
                if trade.get("quantity") is not None
                else trade.get("size")
                if trade.get("size") is not None
                else trade.get("contracts")
            )
            try:
                qty = float(qty_raw) if qty_raw is not None else 1.0
            except Exception:
                qty = 1.0

            all_rows.append({"ts": ts, "price": price, "qty": qty})

        next_cursor = payload.get("cursor")
        if not next_cursor:
            break
        cursor = str(next_cursor)

    if not all_rows:
        return empty

    df = pd.DataFrame(all_rows, columns=columns)
    df = df.dropna(subset=["ts", "price"]).drop_duplicates().sort_values("ts").reset_index(drop=True)
    return df if not df.empty else empty


def get_trades(ticker: str, start_ts: int, end_ts: int) -> pd.DataFrame:
    """Fetch trades; responses cached briefly per ticker + 30s-bucketed window."""
    cache_key = f"{ticker}_{int(start_ts) // 30}_{int(end_ts) // 30}"
    with _cache_lock:
        now = time.time()
        cached = _trade_cache.get(cache_key)
        if cached and now - float(cached.get("ts") or 0) < TRADE_CACHE_TTL:
            df = cached.get("data")
            if isinstance(df, pd.DataFrame):
                return df.copy()

    result = _fetch_trades(ticker, start_ts, end_ts)

    with _cache_lock:
        _trade_cache[cache_key] = {"data": result, "ts": time.time()}
        if len(_trade_cache) > _MAX_TRADE_CACHE_ENTRIES:
            oldest = min(_trade_cache, key=lambda k: float(_trade_cache[k].get("ts") or 0))
            del _trade_cache[oldest]
    return result.copy() if isinstance(result, pd.DataFrame) else result


def get_market_resolution(ticker: str) -> dict | None:
    """Fetch market resolution metadata for a ticker."""
    url = f"{BASE_URL}/markets/{ticker}"
    payload = _get(url, None)
    if payload is None:
        return None

    market = payload.get("market") if isinstance(payload.get("market"), dict) else payload
    if not isinstance(market, dict):
        logger.error("Unexpected market payload shape for ticker=%s", ticker)
        return None

    status = market.get("status")
    result = market.get("result")
    result_str = None
    if isinstance(result, str):
        lowered = result.strip().lower()
        if lowered in {"yes", "no"}:
            result_str = lowered

    final_price = _normalize_price(
        market.get("resolution_price")
        if market.get("resolution_price") is not None
        else market.get("close_price")
        if market.get("close_price") is not None
        else market.get("yes_price")
    )

    resolved = bool((isinstance(status, str) and status.lower() == "finalized") or (result_str is not None))

    return {
        "ticker": market.get("ticker", ticker),
        "resolved": resolved,
        "result": result_str,
        "final_price": final_price,
    }


def get_btc_price() -> float | None:
    """Return cached live BTC spot price in USD from CoinGecko."""
    global _btc_price_cache, _btc_429_until
    now = time.time()
    if now < _btc_429_until:
        return _btc_price_cache.get("price")
    if (now - float(_btc_price_cache.get("ts") or 0)) < BTC_PRICE_CACHE_TTL:
        return _btc_price_cache.get("price")
    try:
        response = requests.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={"ids": "bitcoin", "vs_currencies": "usd"},
            timeout=REQUEST_TIMEOUT,
        )
        if response.status_code == 429:
            _btc_429_until = time.time() + 300
            logger.warning("CoinGecko rate limited, backing off 5 min")
            return _btc_price_cache.get("price")
        response.raise_for_status()
        payload = response.json()
        price_raw = payload.get("bitcoin", {}).get("usd") if isinstance(payload, dict) else None
        price = float(price_raw) if price_raw is not None else None
    except Exception:
        logger.exception("Failed to fetch BTC price from CoinGecko")
        return _btc_price_cache.get("price")
    _btc_price_cache = {"price": price, "ts": now}
    return price


def get_market_prices(ticker: str) -> dict[str, Any] | None:
    """Fetch market quote fields and derive YES/NO bid/ask prices."""
    if not ticker:
        return None
    payload = _get(f"{BASE_URL}/markets/{ticker}", None)
    if payload is None:
        return None
    market = payload.get("market") if isinstance(payload.get("market"), dict) else payload
    if not isinstance(market, dict):
        return None

    def _pick(*keys: str) -> float | None:
        for key in keys:
            if key in market and market.get(key) is not None:
                val = _normalize_price(market.get(key))
                if val is not None:
                    return val
        return None

    yes_bid = _pick("yes_bid", "best_yes_bid", "yes_bid_price")
    yes_ask = _pick("yes_ask", "best_yes_ask", "yes_ask_price")
    last_price = _pick("last_price", "last_price_dollars", "yes_price", "close_price")
    if yes_bid is None and last_price is not None:
        yes_bid = last_price
    if yes_ask is None and last_price is not None:
        yes_ask = last_price
    if yes_bid is None or yes_ask is None:
        return None
    yes_bid = max(0.0, min(1.0, yes_bid))
    yes_ask = max(0.0, min(1.0, yes_ask))
    no_bid = max(0.0, min(1.0, 1.0 - yes_ask))
    no_ask = max(0.0, min(1.0, 1.0 - yes_bid))

    volume_raw = market.get("volume") if market.get("volume") is not None else market.get("total_volume")
    try:
        volume = int(float(volume_raw)) if volume_raw is not None else None
    except Exception:
        volume = None

    return {
        "yes_bid": yes_bid,
        "yes_ask": yes_ask,
        "no_bid": no_bid,
        "no_ask": no_ask,
        "last_price": last_price,
        "volume": volume,
        "ticker": market.get("ticker", ticker),
    }


