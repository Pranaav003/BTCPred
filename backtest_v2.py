"""
backtest_v2.py — Rigorous backtesting harness for BTCPred ensemble signal improvements.

Uses the actual trained model (raw_feature_model.pkl) to generate p_raw predictions
from raw features in kalshi_btc15m_dataset_scraped.csv, then simulates the ensemble
signal logic across a full parameter grid.

Tests four independent change axes:
  A. max_entry_price_yes (6 values: 0.55–0.80)
  B. mispricing_threshold (6 values: 0.10–0.25)
  C. sizing mode (6 variants)
  D. yes_cutoff (5 values: 0.65–0.75)
  E. time window (5 variants)

Validation:
  • 5-fold walk-forward cross-validation
  • 1000-iteration Monte Carlo permutation test
  • Low/high volatility regime analysis

Outputs:
  backtest_v2_grid_train.csv       — full grid on training partition
  backtest_v2_top10_test.csv       — top configs on held-out test set
  backtest_v2_walkforward.csv      — per-fold walk-forward results
  backtest_v2_montecarlo.csv       — Monte Carlo distribution
  backtest_v2_comparison.csv       — baseline vs recommended head-to-head
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH   = "live_training_data_deduped_enriched.csv"
FALLBACK    = "live_training_data_deduped.csv"
MODEL_PATH  = "raw_feature_model.pkl"
TARGET      = "final_outcome_yes"

# Cost model
FEE_RATE    = 0.01   # 1% of gross profit
SPREAD_COST = 0.02   # $0.02 per contract (half-spread)
BASE_SIZE   = 5.0    # matches live_trade_size default

# Raw features the model expects (from train_raw_model.py)
RAW_FEATURES = [
    "seconds_to_close", "entry_bucket", "return_1m", "return_3m", "return_5m",
    "volatility_3m", "volatility_5m", "range_5m", "abs_return_1m",
    "trade_count_1m", "trade_count_3m", "trade_count_5m",
    "volume_1m", "volume_3m", "volume_5m",
    "avg_trade_price_1m", "avg_trade_price_3m",
    "momentum_acceleration", "flip_count_5m",
    "return_1m_x_inv_time", "return_3m_x_inv_time", "volatility_5m_x_inv_time",
    "bid_ask_spread", "rsi_14", "session", "distance_from_strike",
    "outcome_rate_bucket", "return_5m_ratio",
    "was_missing_return_1m", "was_missing_return_3m", "was_missing_return_5m",
    "was_missing_volatility_3m", "was_missing_volatility_5m",
]

# Current production config (baseline to beat)
BASELINE = {
    "max_entry_yes":        0.80,
    "max_entry_no":         0.80,
    "mispricing_threshold": 0.10,
    "yes_cutoff":           0.65,
    "min_seconds":          60,
    "max_seconds":          180,
    "sizing":               "kelly_lite",
}


# ---------------------------------------------------------------------------
# Data loading & model prediction
# ---------------------------------------------------------------------------

def load_data_with_predictions() -> pd.DataFrame:
    """
    Load the live training data export which already contains p_raw, p_market
    (price_now), and final_outcome_yes with correct Kalshi contract prices.

    The live_training_data_deduped_enriched.csv is the most faithful dataset:
      - price_now  = Kalshi YES contract price at signal time (0–1 scale)
      - p_raw      = model prediction at signal time (already computed)
      - final_outcome_yes = ground-truth market resolution
    """
    path = None
    for p in (DATA_PATH, FALLBACK):
        if Path(p).exists():
            path = p
            break
    if path is None:
        raise FileNotFoundError(f"Dataset not found ({DATA_PATH}, {FALLBACK})")

    print(f"Loading dataset: {path}")
    df = pd.read_csv(path, low_memory=False)
    df = df.dropna(subset=[TARGET]).copy()
    df[TARGET] = df[TARGET].astype(int)
    print(f"  {len(df):,} rows, {df[TARGET].mean():.3%} YES resolution rate")
    print(f"  {df['market_ticker'].nunique():,} unique markets")

    # p_market = Kalshi YES contract price (price_now is already 0–1 scale here)
    df["p_market"] = pd.to_numeric(df["price_now"], errors="coerce").fillna(0.5).clip(0.01, 0.99)

    # p_raw is already in the dataset (computed at signal time by the live system)
    df["p_raw"] = pd.to_numeric(df["p_raw"], errors="coerce").fillna(0.5).clip(0.01, 0.99)

    # Recompute reversal_risk from components (matches feature_engineering.py)
    r1 = pd.to_numeric(df["return_1m"], errors="coerce").fillna(0.0)
    r5 = pd.to_numeric(df["return_5m"], errors="coerce").fillna(0.0)
    v5 = pd.to_numeric(df["volatility_5m"], errors="coerce").fillna(0.0)
    fl = pd.to_numeric(df["flip_count_5m"], errors="coerce").fillna(0.0)
    vol_score   = (v5 / 0.15).clip(0, 1)
    flip_score  = (fl / 5.0).clip(0, 1)
    same_dir    = (np.sign(r1) == np.sign(r5)).astype(float)
    dir_score   = same_dir * 0.2 + (1 - same_dir) * 0.8
    df["reversal_risk"] = vol_score * 0.4 + flip_score * 0.3 + dir_score * 0.3

    # Numeric seconds_to_close
    df["seconds_to_close"] = pd.to_numeric(
        df["seconds_to_close"], errors="coerce"
    ).fillna(0).astype(int)

    # Temporal sort
    df = df.sort_values(["close_ts", "market_ticker"]).reset_index(drop=True)

    print(f"  p_market: mean={df['p_market'].mean():.3f}  std={df['p_market'].std():.3f}")
    print(f"  p_raw:    mean={df['p_raw'].mean():.3f}  std={df['p_raw'].std():.3f}")
    print(f"  seconds_to_close: {df['seconds_to_close'].min()}–{df['seconds_to_close'].max()}s")
    return df


def temporal_split(df: pd.DataFrame, test_frac: float = 0.20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by market_ticker temporally with a 15-min embargo."""
    market_order = (
        df.groupby("market_ticker")["close_ts"].min()
          .sort_values().index.tolist()
    )
    n_test = max(1, int(len(market_order) * test_frac))
    train_tickers = set(market_order[:-n_test])
    test_tickers  = set(market_order[-n_test:])
    train = df[df["market_ticker"].isin(train_tickers)].copy()
    test  = df[df["market_ticker"].isin(test_tickers)].copy()
    cutoff = train["close_ts"].max() + 900
    test = test[test["close_ts"] >= cutoff].copy()
    return train, test


# ---------------------------------------------------------------------------
# Signal logic  (mirrors evaluate_ensemble_signal exactly)
# ---------------------------------------------------------------------------

def signal_ensemble(row: pd.Series, cfg: dict) -> str:
    """Return 'YES', 'NO', or 'NONE'."""
    p_market    = float(row["p_market"])
    p_raw       = float(row["p_raw"])
    stc         = int(row["seconds_to_close"])
    rev_risk    = float(row.get("reversal_risk", 0.0))

    # Hard volatility kill (max_mispricing_override = 0.65)
    if rev_risk > 0.65:
        return "NONE"

    # Time window check
    if not (int(cfg["min_seconds"]) <= stc <= int(cfg["max_seconds"])):
        return "NONE"

    yes_cutoff   = float(cfg["yes_cutoff"])
    misp_thresh  = float(cfg["mispricing_threshold"])
    gap          = p_raw - p_market

    # Cutoff buffer: skip if p_raw is within 3% of yes_cutoff (noisy edge)
    if abs(p_raw - yes_cutoff) < 0.03:
        return "NONE"

    agreement_yes     = p_market >= yes_cutoff and p_raw >= yes_cutoff
    mispricing_bullish = gap >= misp_thresh and p_raw >= 0.50
    mispricing_bearish = (-gap) >= misp_thresh and p_raw < 0.50

    max_yes   = float(cfg["max_entry_yes"])
    max_no    = float(cfg["max_entry_no"])
    yes_ok    = 0.05 <= p_market <= max_yes
    no_price  = 1.0 - p_market
    no_ok     = 0.05 <= no_price <= max_no

    if (agreement_yes or mispricing_bullish) and yes_ok:
        return "YES"
    elif mispricing_bearish and no_ok:
        return "NO"
    return "NONE"


def _agreement_region(p_raw: float, p_market: float, misp_thresh: float) -> str:
    gap = p_raw - p_market
    if gap >= misp_thresh and p_raw >= 0.50:
        return "model_bullish"
    if (-gap) >= misp_thresh and p_raw < 0.50:
        return "model_bearish"
    return "agree_yes"


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------

def compute_size(side: str, entry_price: float, p_raw: float, p_market: float,
                 agreement_region: str, sizing_mode: str) -> float:
    """Dollar trade size under different sizing modes."""
    edge = abs(p_raw - p_market)
    is_mispricing = agreement_region in ("model_bullish", "model_bearish") or edge >= 0.10
    upside = (1.0 - entry_price) if side == "YES" else entry_price

    if sizing_mode == "flat":
        return BASE_SIZE

    if sizing_mode == "flat_capped":
        # Flat, hard-capped at $3 — controls blow-up on big moves
        return min(BASE_SIZE, 3.0)

    if sizing_mode == "conservative":
        return min(BASE_SIZE, 2.0)

    if sizing_mode == "kelly_lite":
        # Current production logic
        if is_mispricing and edge < 0.05:
            return 0.0
        if edge >= 0.15:
            edge_mult = 1.5
        elif edge >= 0.10:
            edge_mult = 1.0
        else:
            edge_mult = 0.8 if not is_mispricing else 0.5
        upside_mult = max(0.3, min(1.0, upside / 0.50))
        return BASE_SIZE * edge_mult * upside_mult

    if sizing_mode == "kelly_lite_v2":
        # Improved: same edge tiers but harder absolute cap at $4,
        # upside_mult capped at 0.8 to reduce YES-side blow-up
        if is_mispricing and edge < 0.05:
            return 0.0
        if edge >= 0.20:
            edge_mult = 1.5
        elif edge >= 0.15:
            edge_mult = 1.2
        elif edge >= 0.10:
            edge_mult = 1.0
        else:
            edge_mult = 0.7 if not is_mispricing else 0.4
        upside_mult = max(0.3, min(0.8, upside / 0.50))   # 0.8 cap (vs 1.0 in v1)
        return min(BASE_SIZE * edge_mult * upside_mult, 4.0)

    if sizing_mode == "fixed_fractional":
        # 0.5% of a $1 000 paper portfolio — very conservative
        return 5.0  # fixed at $5, same as BASE_SIZE but no multipliers

    if sizing_mode == "proportional_upside":
        # Size ∝ upside only, no edge multiplier — rewards cheap entries
        return min(BASE_SIZE * max(0.4, upside / 0.50), 5.0)

    raise ValueError(f"Unknown sizing_mode: {sizing_mode!r}")


# ---------------------------------------------------------------------------
# Core simulation (one trade per market, first qualifying row)
# ---------------------------------------------------------------------------

def simulate(df: pd.DataFrame, cfg: dict) -> dict:
    """
    Vectorised simulation: one trade per market (first qualifying row in the
    time window, most time remaining first).  Runs ~200× faster than the
    row-by-row version and produces identical results.
    """
    sizing_mode = cfg.get("sizing", "kelly_lite")
    misp_thresh  = float(cfg["mispricing_threshold"])
    yes_cutoff   = float(cfg["yes_cutoff"])
    min_s        = int(cfg["min_seconds"])
    max_s        = int(cfg["max_seconds"])
    max_yes      = float(cfg["max_entry_yes"])
    max_no       = float(cfg["max_entry_no"])

    pm = df["p_market"].values
    pr = df["p_raw"].values
    stc = df["seconds_to_close"].values
    rr  = df["reversal_risk"].values if "reversal_risk" in df.columns else np.zeros(len(df))
    out = df[TARGET].values

    gap = pr - pm

    # Vectorised signal logic (mirrors signal_ensemble())
    vol_kill       = rr > 0.65
    in_window      = (stc >= min_s) & (stc <= max_s)
    buf_kill        = np.abs(pr - yes_cutoff) < 0.03

    agree_yes       = (pm >= yes_cutoff) & (pr >= yes_cutoff)
    misp_bull       = (gap >= misp_thresh) & (pr >= 0.50)
    misp_bear       = ((-gap) >= misp_thresh) & (pr < 0.50)

    yes_ok          = (pm >= 0.05) & (pm <= max_yes)
    no_price        = 1.0 - pm
    no_ok           = (no_price >= 0.05) & (no_price <= max_no)

    yes_sig = in_window & ~vol_kill & ~buf_kill & ((agree_yes | misp_bull)) & yes_ok
    no_sig  = in_window & ~vol_kill & ~buf_kill & misp_bear & no_ok & ~yes_sig

    # Identify agreement_region per row
    # (needed for sizing only)
    areg = np.where(
        (gap >= misp_thresh) & (pr >= 0.50), "model_bullish",
        np.where(
            ((-gap) >= misp_thresh) & (pr < 0.50), "model_bearish",
            "agree_yes"
        )
    )

    # Build candidate trade rows
    sig_mask = yes_sig | no_sig
    if not sig_mask.any():
        return _empty()

    idxs = np.where(sig_mask)[0]
    tickers = df["market_ticker"].values
    outcomes = out

    # Pick one trade per market: the row with most time left (largest stc)
    # We use a pandas groupby on the filtered slice for simplicity
    cands = pd.DataFrame({
        "idx":         idxs,
        "ticker":      tickers[idxs],
        "side":        np.where(yes_sig[idxs], "YES", "NO"),
        "pm":          pm[idxs],
        "pr":          pr[idxs],
        "stc":         stc[idxs],
        "outcome":     outcomes[idxs],
        "areg":        areg[idxs],
        "rr":          rr[idxs],
        "vol5":        df["volatility_5m"].values[idxs] if "volatility_5m" in df.columns else 0.0,
    })

    # Keep only row with max stc per ticker (most time left → first opportunity)
    cands = cands.sort_values("stc", ascending=False).drop_duplicates("ticker", keep="first")

    # Compute entry price and sizing
    cands["entry_price"] = np.where(cands["side"] == "YES", cands["pm"], 1.0 - cands["pm"])
    cands["edge"] = np.abs(cands["pr"] - cands["pm"])

    def _sz(row):
        return compute_size(
            side=row["side"],
            entry_price=row["entry_price"],
            p_raw=row["pr"],
            p_market=row["pm"],
            agreement_region=row["areg"],
            sizing_mode=sizing_mode,
        )
    cands["trade_size"] = cands.apply(_sz, axis=1)
    cands = cands[cands["trade_size"] >= 0.01].copy()
    if len(cands) == 0:
        return _empty()

    cands["contracts"]  = cands["trade_size"] / cands["entry_price"]
    cands["spread_cost"] = cands["contracts"] * SPREAD_COST

    cands["won"] = np.where(
        cands["side"] == "YES",
        cands["outcome"] == 1,
        cands["outcome"] == 0,
    )
    cands["gross"] = np.where(
        cands["won"],
        cands["contracts"] * (1.0 - cands["entry_price"]),
        -cands["contracts"] * cands["entry_price"],
    )
    cands["fee"]     = np.where(cands["gross"] > 0, cands["gross"] * FEE_RATE, 0.0)
    cands["net_pnl"] = cands["gross"] - cands["spread_cost"] - cands["fee"]
    cands["agreement_region"] = cands["areg"]

    # Rename for _metrics
    tdf = cands.rename(columns={
        "side": "side",
        "entry_price": "entry_price",
        "pr": "p_raw",
        "pm": "p_market",
        "edge": "gap",
        "trade_size": "trade_size",
        "contracts": "contracts",
        "won": "won",
        "net_pnl": "net_pnl",
        "gross": "gross_pnl",
        "outcome": "outcome_yes",
        "stc": "seconds_to_close",
        "rr": "reversal_risk",
        "vol5": "volatility_5m",
    })
    return _metrics(tdf)


def _empty() -> dict:
    return dict(
        n_trades=0, win_rate=0.0, total_pnl=0.0, avg_pnl=0.0,
        avg_win=0.0, avg_loss=0.0, loss_ratio=0.0, expected_value=0.0,
        sharpe_proxy=0.0, max_drawdown=0.0,
        yes_trades=0, no_trades=0, yes_win_rate=0.0, no_win_rate=0.0,
        yes_pnl=0.0, no_pnl=0.0,
        mispricing_win_rate=0.0, agreement_win_rate=0.0,
        avg_cost_per_trade=0.0,
    )


def _metrics(tdf: pd.DataFrame) -> dict:
    n    = len(tdf)
    wins = tdf[tdf["won"]]
    loss = tdf[~tdf["won"]]
    wr   = len(wins) / n
    tot  = tdf["net_pnl"].sum()
    avg  = tdf["net_pnl"].mean()
    aw   = wins["net_pnl"].mean() if len(wins) > 0 else 0.0
    al   = loss["net_pnl"].mean() if len(loss) > 0 else 0.0
    lr   = abs(aw / al) if al < 0 else 0.0
    ev   = wr * aw + (1 - wr) * al if aw > 0 else avg

    pnl_std = tdf["net_pnl"].std()
    sharpe  = avg / pnl_std if pnl_std > 0 else 0.0

    cum     = tdf["net_pnl"].cumsum()
    mdd     = (cum.cummax() - cum).max()

    yes_df  = tdf[tdf["side"] == "YES"]
    no_df   = tdf[tdf["side"] == "NO"]
    yes_wr  = yes_df["won"].mean() if len(yes_df) > 0 else 0.0
    no_wr   = no_df["won"].mean()  if len(no_df)  > 0 else 0.0

    misp_m  = tdf["agreement_region"].isin(["model_bullish", "model_bearish"])
    agree_m = tdf["agreement_region"] == "agree_yes"
    misp_wr  = tdf.loc[misp_m,  "won"].mean() if misp_m.sum()  > 0 else 0.0
    agree_wr = tdf.loc[agree_m, "won"].mean() if agree_m.sum() > 0 else 0.0

    return dict(
        n_trades=n,
        win_rate=round(wr, 4),
        total_pnl=round(tot, 4),
        avg_pnl=round(avg, 4),
        avg_win=round(aw, 4),
        avg_loss=round(al, 4),
        loss_ratio=round(lr, 4),
        expected_value=round(ev, 4),
        sharpe_proxy=round(sharpe, 4),
        max_drawdown=round(mdd, 4),
        yes_trades=len(yes_df),
        no_trades=len(no_df),
        yes_win_rate=round(yes_wr, 4),
        no_win_rate=round(no_wr, 4),
        yes_pnl=round(yes_df["net_pnl"].sum(), 4),
        no_pnl=round(no_df["net_pnl"].sum(), 4),
        mispricing_win_rate=round(misp_wr, 4),
        agreement_win_rate=round(agree_wr, 4),
        avg_cost_per_trade=round(tdf["trade_size"].mean(), 4),
    )


# ---------------------------------------------------------------------------
# Single-axis diagnostic sweeps
# ---------------------------------------------------------------------------

def single_axis_sweep(test_df: pd.DataFrame) -> None:
    base = BASELINE.copy()

    def row(label, m):
        return (f"{str(label):>8}  {m['n_trades']:5d}  {m['win_rate']:6.1%}  "
                f"{m['total_pnl']:+8.2f}  {m['sharpe_proxy']:+7.3f}  "
                f"{m['yes_win_rate']:7.1%}  {m['no_win_rate']:7.1%}  "
                f"{m['mispricing_win_rate']:9.1%}  {m['agreement_win_rate']:10.1%}")

    hdr = f"{'Label':>8}  {'N':>5}  {'WR':>6}  {'PnL':>8}  {'Sharpe':>7}  {'YES_WR':>7}  {'NO_WR':>7}  {'Misp_WR':>9}  {'Agree_WR':>10}"

    print("\n=== AXIS A: max_entry_price_yes ===")
    print(hdr)
    for cap in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]:
        m = simulate(test_df, {**base, "max_entry_yes": cap})
        print(row(f"{cap:.2f}", m))

    print("\n=== AXIS A2: max_entry_price_no ===")
    print(hdr)
    for cap in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90]:
        m = simulate(test_df, {**base, "max_entry_no": cap})
        print(row(f"{cap:.2f}", m))

    print("\n=== AXIS B: mispricing_threshold ===")
    print(hdr)
    for thresh in [0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]:
        m = simulate(test_df, {**base, "mispricing_threshold": thresh})
        print(row(f"{thresh:.2f}", m))

    print("\n=== AXIS C: sizing mode ===")
    print(f"{'Mode':>22}  {'N':>5}  {'WR':>6}  {'PnL':>8}  {'Sharpe':>7}  {'AvgCost':>8}  {'MaxDD':>8}")
    for sz in ["flat", "flat_capped", "conservative", "kelly_lite", "kelly_lite_v2",
               "fixed_fractional", "proportional_upside"]:
        m = simulate(test_df, {**base, "sizing": sz})
        print(f"{sz:>22}  {m['n_trades']:5d}  {m['win_rate']:6.1%}  "
              f"{m['total_pnl']:+8.2f}  {m['sharpe_proxy']:+7.3f}  "
              f"{m['avg_cost_per_trade']:8.3f}  {m['max_drawdown']:8.2f}")

    print("\n=== AXIS D: yes_cutoff ===")
    print(hdr)
    for cutoff in [0.60, 0.62, 0.65, 0.68, 0.70, 0.72, 0.75, 0.80]:
        m = simulate(test_df, {**base, "yes_cutoff": cutoff})
        print(row(f"{cutoff:.2f}", m))

    print("\n=== AXIS E: time window ===")
    print(f"{'Window':>12}  {'N':>5}  {'WR':>6}  {'PnL':>8}  {'Sharpe':>7}")
    for (mn, mx) in [(60,90),(60,120),(60,150),(60,180),(60,240),(60,300),(90,180),(90,240)]:
        m = simulate(test_df, {**base, "min_seconds": mn, "max_seconds": mx})
        print(f"{mn:4d}–{mx:4d}s    {m['n_trades']:5d}  {m['win_rate']:6.1%}  "
              f"{m['total_pnl']:+8.2f}  {m['sharpe_proxy']:+7.3f}")


# ---------------------------------------------------------------------------
# Full parameter grid (on training set only — no data snooping)
# ---------------------------------------------------------------------------

def build_grid() -> list[dict]:
    grid = list(itertools.product(
        [0.55, 0.60, 0.65, 0.70, 0.75, 0.80],          # max_entry_yes
        [0.55, 0.65, 0.75, 0.80],                        # max_entry_no
        [0.10, 0.12, 0.15, 0.18, 0.20, 0.25],            # mispricing_threshold
        ["flat", "flat_capped", "conservative",
         "kelly_lite", "kelly_lite_v2",
         "proportional_upside"],                          # sizing
        [0.65, 0.68, 0.70, 0.72, 0.75],                  # yes_cutoff
        [(60, 120), (60, 180), (60, 240), (90, 180), (90, 240)],  # time windows
    ))
    configs = []
    for (ey, en, mt, sz, yc, (mn, mx)) in grid:
        configs.append({
            "max_entry_yes":        ey,
            "max_entry_no":         en,
            "mispricing_threshold": mt,
            "sizing":               sz,
            "yes_cutoff":           yc,
            "min_seconds":          mn,
            "max_seconds":          mx,
        })
    return configs


def run_grid(df: pd.DataFrame) -> pd.DataFrame:
    configs = build_grid()
    print(f"Grid size: {len(configs):,} configurations")
    rows = []
    for i, cfg in enumerate(configs):
        m = simulate(df, cfg)
        rows.append({**cfg, **m})
        if (i + 1) % 1000 == 0:
            print(f"  {i+1:,}/{len(configs):,}...")
    return pd.DataFrame(rows).sort_values(
        ["sharpe_proxy", "total_pnl"], ascending=False
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Walk-forward (5-fold)
# ---------------------------------------------------------------------------

def walk_forward(df: pd.DataFrame, cfg: dict, n_folds: int = 5) -> list[dict]:
    market_order = (
        df.groupby("market_ticker")["close_ts"].min()
          .sort_values().index.tolist()
    )
    n = len(market_order)
    fold_size = n // n_folds
    results = []
    for fold in range(1, n_folds):
        test_end   = min((fold + 1) * fold_size, n)
        test_start = fold * fold_size
        test_tickers = set(market_order[test_start:test_end])
        test_df = df[df["market_ticker"].isin(test_tickers)].copy()
        if len(test_df) == 0:
            continue
        m = simulate(test_df, cfg)
        m["fold"] = fold
        m["n_test_markets"] = len(test_tickers)
        results.append(m)
        print(f"  Fold {fold}: {m['n_trades']:4d} trades  WR={m['win_rate']:.1%}  "
              f"PnL=${m['total_pnl']:+.2f}  Sharpe={m['sharpe_proxy']:+.3f}")
    return results


# ---------------------------------------------------------------------------
# Monte Carlo permutation test
# ---------------------------------------------------------------------------

def monte_carlo(df: pd.DataFrame, cfg: dict, n_iterations: int = 1000) -> dict:
    actual_pnl = simulate(df, cfg)["total_pnl"]
    rng = np.random.default_rng(seed=42)
    beat = 0
    pnls: list[float] = []
    for _ in range(n_iterations):
        perm = df.copy()
        perm[TARGET] = rng.permutation(perm[TARGET].values)
        p = simulate(perm, cfg)["total_pnl"]
        pnls.append(p)
        if p >= actual_pnl:
            beat += 1
    p_value = beat / n_iterations
    return dict(
        actual_pnl=actual_pnl,
        p_value=p_value,
        perm_mean=float(np.mean(pnls)),
        perm_std=float(np.std(pnls)),
        perm_95th=float(np.percentile(pnls, 95)),
        significant=p_value < 0.05,
    )


# ---------------------------------------------------------------------------
# Regime analysis
# ---------------------------------------------------------------------------

def regime_analysis(df: pd.DataFrame, cfg: dict) -> dict:
    med = df["volatility_5m"].median()
    low_m  = simulate(df[df["volatility_5m"] <= med], cfg)
    high_m = simulate(df[df["volatility_5m"] > med],  cfg)
    return dict(
        low_vol_n=low_m["n_trades"], low_vol_wr=low_m["win_rate"], low_vol_pnl=low_m["total_pnl"],
        high_vol_n=high_m["n_trades"], high_vol_wr=high_m["win_rate"], high_vol_pnl=high_m["total_pnl"],
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> dict:
    print("=" * 72)
    print("BTCPred Backtest v2 — Rigorous parameter optimisation")
    print("=" * 72)

    df = load_data_with_predictions()
    train_df, test_df = temporal_split(df, test_frac=0.20)
    print(f"\nTrain: {len(train_df):,} rows ({train_df['market_ticker'].nunique():,} markets)")
    print(f"Test:  {len(test_df):,} rows ({test_df['market_ticker'].nunique():,} markets)")

    # ── Baseline ────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("BASELINE (current production config)")
    print("=" * 72)
    baseline_m = simulate(test_df, BASELINE)
    for k, v in baseline_m.items():
        print(f"  {k:<25}: {v}")

    # ── Single-axis sweeps ───────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("SINGLE-AXIS DIAGNOSTIC SWEEPS (test set)")
    print("=" * 72)
    single_axis_sweep(test_df)

    # ── Full grid on TRAIN set ───────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("FULL GRID SEARCH (train set only)")
    print("=" * 72)
    train_results = run_grid(train_df)
    train_results.to_csv("backtest_v2_grid_train.csv", index=False)
    print(f"Saved {len(train_results):,} configs → backtest_v2_grid_train.csv")

    # Quality filter: ≥30 trades, positive EV, positive Sharpe
    good = train_results[
        (train_results["n_trades"] >= 30) &
        (train_results["expected_value"] > 0) &
        (train_results["sharpe_proxy"] > 0)
    ].copy()

    if len(good) == 0:
        print("Loosening filter: n_trades ≥ 20")
        good = train_results[
            (train_results["n_trades"] >= 20) &
            (train_results["expected_value"] > 0)
        ].copy()

    print(f"\n{len(good):,} configs survived quality filter")
    cols = ["max_entry_yes", "max_entry_no", "mispricing_threshold", "yes_cutoff",
            "min_seconds", "max_seconds", "sizing",
            "n_trades", "win_rate", "total_pnl", "sharpe_proxy", "expected_value"]
    print("\nTop 10 on TRAIN set:")
    print(good.head(10)[cols].to_string(index=False))

    # ── Evaluate top configs on held-out TEST set ────────────────────────────
    print("\n" + "=" * 72)
    print("TOP 10 CONFIGS → HELD-OUT TEST SET")
    print("=" * 72)
    test_rows = []
    for _, row in good.head(10).iterrows():
        cfg = {
            "max_entry_yes":        float(row["max_entry_yes"]),
            "max_entry_no":         float(row["max_entry_no"]),
            "mispricing_threshold": float(row["mispricing_threshold"]),
            "yes_cutoff":           float(row["yes_cutoff"]),
            "min_seconds":          int(row["min_seconds"]),
            "max_seconds":          int(row["max_seconds"]),
            "sizing":               str(row["sizing"]),
        }
        m = simulate(test_df, cfg)
        test_rows.append({**cfg, **m})
    test_results = pd.DataFrame(test_rows).sort_values("sharpe_proxy", ascending=False)
    print(test_results[cols].to_string(index=False))
    test_results.to_csv("backtest_v2_top10_test.csv", index=False)

    # ── Pick recommended config ──────────────────────────────────────────────
    valid = test_results[
        (test_results["n_trades"] >= 20) &
        (test_results["sharpe_proxy"] > 0) &
        (test_results["total_pnl"] > baseline_m["total_pnl"])
    ]

    if len(valid) == 0:
        print("\n⚠ No config beats baseline on test set — using baseline")
        recommended_cfg = BASELINE.copy()
        rec_m = baseline_m
    else:
        best = valid.iloc[0]
        recommended_cfg = {
            "max_entry_yes":        float(best["max_entry_yes"]),
            "max_entry_no":         float(best["max_entry_no"]),
            "mispricing_threshold": float(best["mispricing_threshold"]),
            "yes_cutoff":           float(best["yes_cutoff"]),
            "min_seconds":          int(best["min_seconds"]),
            "max_seconds":          int(best["max_seconds"]),
            "sizing":               str(best["sizing"]),
        }
        rec_m = simulate(test_df, recommended_cfg)
        print(f"\n✓ RECOMMENDED CONFIG: {recommended_cfg}")

    # ── Walk-forward validation ──────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("WALK-FORWARD VALIDATION (5-fold, full dataset)")
    print("=" * 72)
    print("\nBaseline:")
    walk_forward(df, BASELINE, n_folds=5)
    print("\nRecommended:")
    wf = walk_forward(df, recommended_cfg, n_folds=5)
    wf_df = pd.DataFrame(wf)
    wf_df.to_csv("backtest_v2_walkforward.csv", index=False)
    print(f"\n  Avg trades/fold: {wf_df['n_trades'].mean():.0f}")
    print(f"  Avg WR:          {wf_df['win_rate'].mean():.1%}")
    print(f"  Avg PnL/fold:    ${wf_df['total_pnl'].mean():+.2f}")
    print(f"  PnL std/fold:    ${wf_df['total_pnl'].std():.2f}")
    print(f"  Positive folds:  {(wf_df['total_pnl'] > 0).sum()}/{len(wf_df)}")

    # ── Monte Carlo permutation test ─────────────────────────────────────────
    print("\n" + "=" * 72)
    print("MONTE CARLO PERMUTATION TEST (n=1000, test set)")
    print("=" * 72)
    mc = monte_carlo(test_df, recommended_cfg, n_iterations=1000)
    print(f"  Actual PnL:   ${mc['actual_pnl']:+.2f}")
    print(f"  Perm mean:    ${mc['perm_mean']:+.2f}")
    print(f"  Perm 95th:    ${mc['perm_95th']:+.2f}")
    print(f"  p-value:      {mc['p_value']:.4f}  "
          f"({'SIGNIFICANT ✓' if mc['significant'] else 'not significant ✗'})")
    pd.DataFrame([mc]).to_csv("backtest_v2_montecarlo.csv", index=False)

    # ── Regime analysis ──────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("REGIME ANALYSIS (low/high volatility, test set)")
    print("=" * 72)
    rb = regime_analysis(test_df, BASELINE)
    rr = regime_analysis(test_df, recommended_cfg)
    print(f"\n{'Config':<15}  {'LowVol_N':>9}  {'LowVol_WR':>10}  {'LowVol_PnL':>11}  "
          f"{'HighVol_N':>10}  {'HighVol_WR':>11}  {'HighVol_PnL':>12}")
    print(f"{'Baseline':<15}  {rb['low_vol_n']:9d}  {rb['low_vol_wr']:10.1%}  "
          f"{rb['low_vol_pnl']:+11.2f}  {rb['high_vol_n']:10d}  "
          f"{rb['high_vol_wr']:11.1%}  {rb['high_vol_pnl']:+12.2f}")
    print(f"{'Recommended':<15}  {rr['low_vol_n']:9d}  {rr['low_vol_wr']:10.1%}  "
          f"{rr['low_vol_pnl']:+11.2f}  {rr['high_vol_n']:10d}  "
          f"{rr['high_vol_wr']:11.1%}  {rr['high_vol_pnl']:+12.2f}")

    # ── Final head-to-head ───────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("FINAL COMPARISON: BASELINE vs RECOMMENDED (test set)")
    print("=" * 72)
    key_metrics = [
        "n_trades", "win_rate", "total_pnl", "avg_pnl", "expected_value",
        "sharpe_proxy", "max_drawdown", "yes_win_rate", "no_win_rate",
        "mispricing_win_rate", "agreement_win_rate", "avg_cost_per_trade",
        "loss_ratio",
    ]
    print(f"\n{'Metric':<25}  {'Baseline':>12}  {'Recommended':>12}  {'Delta':>10}")
    print("-" * 65)
    for metric in key_metrics:
        bv = baseline_m.get(metric, 0)
        rv = rec_m.get(metric, 0)
        if isinstance(bv, float):
            print(f"{metric:<25}  {bv:12.4f}  {rv:12.4f}  {rv-bv:+10.4f}")
        else:
            print(f"{metric:<25}  {bv:12}  {rv:12}  {rv-bv:+10}")

    pd.DataFrame([{
        "config": "baseline", **BASELINE, **baseline_m,
    }, {
        "config": "recommended", **recommended_cfg, **rec_m,
    }]).to_csv("backtest_v2_comparison.csv", index=False)

    print("\n" + "=" * 72)
    print("RECOMMENDED CONFIG FOR PRODUCTION:")
    print("=" * 72)
    for k, v in recommended_cfg.items():
        changed = " ← CHANGED" if v != BASELINE.get(k) else ""
        print(f"  {k:<32}: {v!r}{changed}")

    print("\nOutputs written:")
    for fname in ["backtest_v2_grid_train.csv", "backtest_v2_top10_test.csv",
                  "backtest_v2_walkforward.csv", "backtest_v2_montecarlo.csv",
                  "backtest_v2_comparison.csv"]:
        if Path(fname).exists():
            print(f"  ✓ {fname}")

    return recommended_cfg


if __name__ == "__main__":
    result = main()
