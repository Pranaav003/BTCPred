"""Non-optimizing evaluation of the CURRENT live signal rule over historical
live-collected data (live_training_data_deduped_enriched.csv).

This is deliberately NOT a parameter search. It measures the realized
per-contract EV of the exact rule the bot runs today, using the model's
real-time (out-of-sample, source='live') predictions and actual outcomes.

Rule (from confirmed /api/settings, aggressive profile):
  - ensemble mispricing: gap = p_raw - price_now
      bullish  YES if gap >= THRESH and p_raw >= 0.50
      bearish  NO  if -gap >= THRESH and p_raw <  0.50
  - entry filters: YES 0.05 <= price_now <= 0.65 ; NO price (1-price_now) 0.05..0.80
  - window: 90 <= seconds_to_close <= 300 (auto-trade effective window)
  - agreement-YES is structurally dead (yes_cutoff 0.70 > max_entry_yes 0.65)
  - one trade per market: first qualifying snapshot (largest seconds_to_close)
Entry cost uses the real aggressive offset (YES +2c; NO +5c if <=40c else +3c).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

# Analysis script: silence pandas 3.0 CoW FutureWarnings (cosmetic; results unaffected).
warnings.filterwarnings("ignore", category=FutureWarning)

CSV = "live_training_data_deduped_enriched.csv"
THRESH = 0.10
MAX_YES = 0.65
MAX_NO = 0.80
MIN_ENTRY = 0.05
WIN_LO, WIN_HI = 90, 300


def entry_cost(side: str, price_now: float) -> float:
    if side == "YES":
        return min(0.99, price_now + 0.02)
    no_price = 1.0 - price_now
    off = 0.05 if no_price <= 0.40 else 0.03
    return min(0.99, no_price + off)


def decide(p_raw: float, price_now: float):
    """Return 'YES' / 'NO' / None for the current mispricing rule + entry filter."""
    gap = p_raw - price_now
    if gap >= THRESH and p_raw >= 0.50:
        if MIN_ENTRY <= price_now <= MAX_YES:
            return "YES"
        return None
    if (-gap) >= THRESH and p_raw < 0.50:
        no_price = 1.0 - price_now
        if MIN_ENTRY <= no_price <= MAX_NO:
            return "NO"
        return None
    return None


def stats(pnls: np.ndarray) -> str:
    n = len(pnls)
    if n == 0:
        return "n=0"
    mean = pnls.mean()
    wr = 100.0 * (pnls > 0).mean()
    if n > 1:
        se = pnls.std(ddof=1) / np.sqrt(n)
        lo, hi = mean - 1.96 * se, mean + 1.96 * se
        ci = f" 95%CI[{lo:+.3f},{hi:+.3f}]{'  (crosses 0)' if lo < 0 < hi else ''}"
    else:
        ci = ""
    return f"n={n:4d}  EV/contract={mean:+.4f}  WR={wr:4.1f}%  total={pnls.sum():+.2f}{ci}"


def main() -> None:
    df = pd.read_csv(CSV)
    df = df[(df["seconds_to_close"] >= WIN_LO) & (df["seconds_to_close"] <= WIN_HI)].copy()
    df["p_raw"] = pd.to_numeric(df["p_raw"], errors="coerce")
    df["price_now"] = pd.to_numeric(df["price_now"], errors="coerce")
    df["final_outcome_yes"] = pd.to_numeric(df["final_outcome_yes"], errors="coerce")
    df = df.dropna(subset=["p_raw", "price_now", "final_outcome_yes"])

    df["decision"] = [decide(p, m) for p, m in zip(df["p_raw"], df["price_now"])]
    sig = df[df["decision"].notna()].copy()

    # One trade per market: earliest qualifying poll (largest seconds_to_close).
    sig = sig.sort_values("seconds_to_close", ascending=False)
    trades = sig.groupby("market_ticker", as_index=False).first()

    def pnl_row(r, use_offset=True):
        price = float(r["price_now"])
        side = r["decision"]
        cost = entry_cost(side, price) if use_offset else (price if side == "YES" else 1.0 - price)
        won = (r["final_outcome_yes"] == 1) if side == "YES" else (r["final_outcome_yes"] == 0)
        return (1.0 if won else 0.0) - cost

    trades["pnl"] = trades.apply(lambda r: pnl_row(r, True), axis=1)
    trades["pnl_mid"] = trades.apply(lambda r: pnl_row(r, False), axis=1)

    n_markets = df["market_ticker"].nunique()
    print(f"markets in window: {n_markets} | qualifying trades (current rule): {len(trades)}")
    print(f"trade frequency: {100*len(trades)/n_markets:.1f}% of markets\n")

    print("=== CURRENT RULE — per-contract PnL (realistic aggressive-offset fills) ===")
    print("ALL      ", stats(trades["pnl"].values))
    print("YES      ", stats(trades[trades.decision == "YES"]["pnl"].values))
    print("NO       ", stats(trades[trades.decision == "NO"]["pnl"].values))
    print("\n=== same, if filled at MID (no offset cost) — optimistic upper bound ===")
    print("ALL      ", stats(trades["pnl_mid"].values))

    print("\n=== NO by entry (NO price) bucket ===")
    t = trades[trades.decision == "NO"].copy()
    t["nop"] = 1.0 - t["price_now"]
    for lo, hi in [(0.05, 0.40), (0.40, 0.60), (0.60, 0.80)]:
        b = t[(t.nop >= lo) & (t.nop < hi)]
        print(f"  NO {lo:.2f}-{hi:.2f}", stats(b["pnl"].values))
    print("=== YES by entry (YES price) bucket ===")
    t = trades[trades.decision == "YES"].copy()
    for lo, hi in [(0.05, 0.45), (0.45, 0.55), (0.55, 0.65)]:
        b = t[(t.price_now >= lo) & (t.price_now < hi)]
        print(f"  YES {lo:.2f}-{hi:.2f}", stats(b["pnl"].values))

    if "session" in trades.columns:
        print("\n=== by session ===")
        for s, g in trades.groupby("session"):
            print(f"  {str(s):10s}", stats(g["pnl"].values))

    print("\n=== threshold sensitivity (INFORMATIONAL — not an optimization target) ===")
    for th in [0.08, 0.10, 0.12, 0.15, 0.20]:
        def dec_th(p, m, th=th):
            g = p - m
            if g >= th and p >= 0.50 and MIN_ENTRY <= m <= MAX_YES:
                return "YES"
            if (-g) >= th and p < 0.50 and MIN_ENTRY <= (1 - m) <= MAX_NO:
                return "NO"
            return None
        d = df.copy()
        d["dec"] = [dec_th(p, m) for p, m in zip(d["p_raw"], d["price_now"])]
        s2 = d[d["dec"].notna()].sort_values("seconds_to_close", ascending=False)
        tr = s2.groupby("market_ticker", as_index=False).first()
        if len(tr):
            pn = tr.apply(lambda r: (1.0 if ((r.final_outcome_yes == 1) if r.dec == "YES" else (r.final_outcome_yes == 0)) else 0.0) - entry_cost(r.dec, float(r.price_now)), axis=1).values
            print(f"  thresh {th:.2f}: ", stats(np.array(pn)))


if __name__ == "__main__":
    main()
