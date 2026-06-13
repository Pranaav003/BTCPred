"""Kalshi trading API authentication via RSA-PSS request signing."""

from __future__ import annotations

import base64
import logging
import os
import time

logger = logging.getLogger(__name__)

TRADING_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
SIGN_PATH_PREFIX = "/trade-api/v2"


def _sign_path(request_path: str) -> str:
    """Full API path used in the signature (no query string)."""
    path = (request_path or "").split("?", 1)[0]
    if not path.startswith("/"):
        path = f"/{path}"
    if path.startswith(SIGN_PATH_PREFIX):
        return path
    return f"{SIGN_PATH_PREFIX}{path}"


def _normalize_pem(pem: str) -> str:
    """Accept full PEM or Render-style body-only paste (no BEGIN/END lines)."""
    pem = pem.replace("\\n", "\n").strip()
    if not pem:
        return ""
    if "BEGIN" not in pem:
        body = "\n".join(line.strip() for line in pem.splitlines() if line.strip())
        pem = f"-----BEGIN RSA PRIVATE KEY-----\n{body}\n-----END RSA PRIVATE KEY-----"
    return pem


def get_private_key():
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import serialization

    pem = _normalize_pem(os.environ.get("KALSHI_PRIVATE_KEY", ""))
    if not pem:
        return None
    try:
        return serialization.load_pem_private_key(
            pem.encode(),
            password=None,
            backend=default_backend(),
        )
    except Exception as exc:
        logger.error("Failed to load Kalshi private key: %s", exc)
        return None


def get_kalshi_headers(method: str, path: str) -> dict | None:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding

    key_id = os.environ.get("KALSHI_API_KEY_ID", "").strip()
    if not key_id:
        return None
    private_key = get_private_key()
    if not private_key:
        return None
    timestamp = str(int(time.time() * 1000))
    sign_path = _sign_path(path)
    message = timestamp + method.upper() + sign_path
    try:
        signature = private_key.sign(
            message.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY": key_id,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode(),
            "Content-Type": "application/json",
        }
    except Exception as exc:
        logger.error("Failed to sign Kalshi request: %s", exc)
        return None


def is_configured() -> bool:
    return bool(
        os.environ.get("KALSHI_API_KEY_ID", "").strip()
        and os.environ.get("KALSHI_PRIVATE_KEY", "").strip()
    )
