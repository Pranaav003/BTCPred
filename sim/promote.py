"""Map a winning search config to live AppSettings keys."""
from __future__ import annotations

# search-config key -> live AppSettings key
_KEY_MAP = {
    "mispricing_threshold": "mispricing_threshold",
    "max_entry_yes": "max_entry_price_yes",
    "max_entry_no": "max_entry_price_no",
    "no_max_p_raw": "no_max_p_raw",
    "yes_cutoff": "yes_cutoff",
}


def config_to_settings(cfg: dict) -> dict:
    out = {}
    for cfg_key, setting_key in _KEY_MAP.items():
        if cfg_key in cfg:
            out[setting_key] = str(cfg[cfg_key])
    return out


def promotion_candidate(board: list):
    passing = [r for r in board if r.get("passed")]
    if not passing:
        return None
    return max(passing, key=lambda r: r["score"])
