"""Load per-poll Kalshi BTC-15m logs into per-market price paths."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

# Columns that are metadata, not model features.
_META_COLS = {
    "market_ticker", "entry_bucket", "close_ts", "seconds_to_close",
    "price_now", "p_raw", "final_outcome_yes", "logged_at", "signal",
    "agreement_region", "source",
}


@dataclass
class Poll:
    seconds_to_close: int
    price_now: float
    p_raw: float
    features: dict = field(default_factory=dict)


@dataclass
class MarketPath:
    ticker: str
    bucket: int
    close_ts: int
    final_outcome_yes: int
    polls: list[Poll]  # ordered by DESCENDING seconds_to_close


def load_paths(csv_path: str, min_polls: int = 1) -> list[MarketPath]:
    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c not in _META_COLS
                    and pd.api.types.is_numeric_dtype(df[c])]
    paths: list = []
    for (ticker, bucket), grp in df.groupby(["market_ticker", "entry_bucket"]):
        grp = grp.sort_values("seconds_to_close", ascending=False)
        if len(grp) < min_polls:
            continue
        polls = [
            Poll(
                seconds_to_close=int(r.seconds_to_close),
                price_now=float(r.price_now),
                p_raw=float(r.p_raw),
                features={c: float(getattr(r, c)) for c in feature_cols
                          if pd.notna(getattr(r, c))},
            )
            for r in grp.itertuples(index=False)
        ]
        first = grp.iloc[0]
        paths.append(MarketPath(
            ticker=str(ticker),
            bucket=int(bucket),
            close_ts=int(first["close_ts"]),
            final_outcome_yes=int(first["final_outcome_yes"]),
            polls=polls,
        ))
    return paths
