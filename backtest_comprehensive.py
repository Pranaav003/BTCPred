"""Comprehensive standalone backtest across all strategy variants."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

DATA_PATH = "kalshi_btc15m_dataset_30k.csv"
MODEL_PATH = "raw_feature_model.pkl"
TARGET = "final_outcome_yes"
TEST_SIZE = 0.20
BASE_TRADE_SIZE = 20.0
FEE_RATE = 0.01  # 1% on profits (matching Kalshi)
SPREAD_COST = 0.02  # 2 cents per contract

RAW_FEATURES = [
    "seconds_to_close",
    "entry_bucket",
    "return_1m",
    "return_3m",
    "return_5m",
    "volatility_3m",
    "volatility_5m",
    "range_5m",
    "abs_return_1m",
    "trade_count_1m",
    "trade_count_3m",
    "trade_count_5m",
    "volume_1m",
    "volume_3m",
    "volume_5m",
    "avg_trade_price_1m",
    "avg_trade_price_3m",
    "momentum_1m",
    "momentum_3m",
    "momentum_5m",
    "momentum_acceleration",
    "price_velocity_5m",
    "flip_count_5m",
    "return_1m_x_inv_time",
    "return_3m_x_inv_time",
    "volatility_5m_x_inv_time",
]

RESULTS_CSV = "backtest_comprehensive_results.csv"
TOP_TRADES_CSV = "backtest_top_trades.csv"

ENTRY_BUCKETS = [0.0, 0.5, 0.65, 0.75, 0.85, 1.01]
ENTRY_BUCKET_LABELS = ["0.00-0.50", "0.50-0.65", "0.65-0.75", "0.75-0.85", "0.85-1.00"]


@dataclass
class Context:
    p_market: np.ndarray
    p_raw: np.ndarray
    seconds: np.ndarray
    outcome_yes: np.ndarray
    row_index: np.ndarray
    n_rows: int


def load_dataset(path: str) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    df = pd.read_csv(csv_path, dtype={"series_ticker": str})
    required = set(RAW_FEATURES + [TARGET, "price_now", "seconds_to_close", "entry_bucket", "close_ts", "market_ticker"])
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df["avg_trade_price_1m"] = df["avg_trade_price_1m"].fillna(df["price_now"])
    df["avg_trade_price_3m"] = df["avg_trade_price_3m"].fillna(df["price_now"])
    df = df.sort_values(by=["close_ts", "market_ticker", "entry_bucket"]).reset_index(drop=True)
    return df


def temporal_test_split(df: pd.DataFrame) -> pd.DataFrame:
    split_idx = int(len(df) * (1 - TEST_SIZE))
    if split_idx <= 0 or split_idx >= len(df):
        raise ValueError("Dataset too small for TEST_SIZE split.")
    return df.iloc[split_idx:].copy().reset_index(drop=True)


def load_model_bundle(path: str) -> tuple[object, list[str]]:
    obj = joblib.load(path)
    if isinstance(obj, dict):
        model = obj.get("model")
        features = obj.get("features") or RAW_FEATURES
        if model is None:
            raise ValueError("MODEL_PATH dict is missing key 'model'.")
        return model, list(features)
    return obj, RAW_FEATURES


def max_drawdown_from_pnls(pnls: np.ndarray) -> float:
    if pnls.size == 0:
        return 0.0
    worst = 0.0
    run = 0.0
    for p in pnls:
        if p < 0:
            run += p
            worst = min(worst, run)
        else:
            run = 0.0
    return abs(float(worst))


def build_trade_df(ctx: Context, mask_yes: np.ndarray, mask_no: np.ndarray, config_tag: str) -> pd.DataFrame:
    records = []
    if np.any(mask_yes):
        idx = np.where(mask_yes)[0]
        entry = ctx.p_market[idx]
        valid = entry > 0
        idx = idx[valid]
        entry = entry[valid]
        contracts = BASE_TRADE_SIZE / entry
        won = (ctx.outcome_yes[idx] == 1).astype(bool)
        gross_pnl = np.where(won, (1.0 - entry) * contracts, -entry * contracts)
        spread_total = SPREAD_COST * contracts
        fee = np.where(won, FEE_RATE * np.maximum(gross_pnl, 0.0), 0.0)
        pnl = gross_pnl - spread_total - fee
        upside = 1.0 - entry
        yes_df = pd.DataFrame(
            {
                "row_idx": idx,
                "side": "YES",
                "entry_price": entry,
                "contracts": contracts,
                "upside": upside,
                "won": won,
                "pnl": pnl,
                "seconds": ctx.seconds[idx],
                "p_market": ctx.p_market[idx],
                "p_raw": ctx.p_raw[idx],
                "outcome_yes": ctx.outcome_yes[idx],
                "config_tag": config_tag,
            }
        )
        records.append(yes_df)
    if np.any(mask_no):
        idx = np.where(mask_no)[0]
        entry = 1.0 - ctx.p_market[idx]
        valid = entry > 0
        idx = idx[valid]
        entry = entry[valid]
        contracts = BASE_TRADE_SIZE / entry
        won = (ctx.outcome_yes[idx] == 0).astype(bool)
        gross_pnl = np.where(won, ctx.p_market[idx] * contracts, -(1.0 - ctx.p_market[idx]) * contracts)
        spread_total = SPREAD_COST * contracts
        fee = np.where(won, FEE_RATE * np.maximum(gross_pnl, 0.0), 0.0)
        pnl = gross_pnl - spread_total - fee
        upside = ctx.p_market[idx]
        no_df = pd.DataFrame(
            {
                "row_idx": idx,
                "side": "NO",
                "entry_price": entry,
                "contracts": contracts,
                "upside": upside,
                "won": won,
                "pnl": pnl,
                "seconds": ctx.seconds[idx],
                "p_market": ctx.p_market[idx],
                "p_raw": ctx.p_raw[idx],
                "outcome_yes": ctx.outcome_yes[idx],
                "config_tag": config_tag,
            }
        )
        records.append(no_df)
    if not records:
        return pd.DataFrame(
            columns=[
                "row_idx",
                "side",
                "entry_price",
                "contracts",
                "upside",
                "won",
                "pnl",
                "seconds",
                "p_market",
                "p_raw",
                "outcome_yes",
                "config_tag",
            ]
        )
    return pd.concat(records, ignore_index=True).sort_values("row_idx").reset_index(drop=True)


def metrics_from_trades(strategy: str, params: str, trades_df: pd.DataFrame) -> dict:
    n = int(len(trades_df))
    if n == 0:
        return {
            "strategy": strategy,
            "params": params,
            "total_trades": 0,
            "yes_trades": 0,
            "no_trades": 0,
            "accuracy": 0.0,
            "yes_accuracy": 0.0,
            "no_accuracy": 0.0,
            "total_pnl": 0.0,
            "avg_pnl": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "loss_ratio": float("inf"),
            "expected_value": 0.0,
            "trades_per_1000_rows": 0.0,
            "max_drawdown": 0.0,
            "sharpe_proxy": 0.0,
            "entry_0_00_0_50_pct": 0.0,
            "entry_0_50_0_65_pct": 0.0,
            "entry_0_65_0_75_pct": 0.0,
            "entry_0_75_0_85_pct": 0.0,
            "entry_0_85_1_00_pct": 0.0,
        }

    won = trades_df["won"].to_numpy(dtype=bool)
    pnl = trades_df["pnl"].to_numpy(dtype=float)
    side_yes = trades_df["side"].to_numpy() == "YES"
    side_no = ~side_yes

    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    avg_win = float(np.mean(wins)) if wins.size else 0.0
    avg_loss = float(np.mean(losses)) if losses.size else 0.0
    accuracy = float(np.mean(won))
    yes_acc = float(np.mean(won[side_yes])) if np.any(side_yes) else 0.0
    no_acc = float(np.mean(won[side_no])) if np.any(side_no) else 0.0
    loss_ratio = abs(avg_loss) / avg_win if avg_win > 0 else float("inf")
    ev = accuracy * avg_win - (1.0 - accuracy) * abs(avg_loss)
    std = float(np.std(pnl))
    sharpe = float(np.mean(pnl) / std) if std > 0 else 0.0

    entry_price = trades_df["entry_price"].to_numpy(dtype=float)
    binned = pd.cut(entry_price, bins=ENTRY_BUCKETS, labels=ENTRY_BUCKET_LABELS, right=False, include_lowest=True)
    dist = pd.Series(binned).value_counts(normalize=True).reindex(ENTRY_BUCKET_LABELS, fill_value=0.0)

    return {
        "strategy": strategy,
        "params": params,
        "total_trades": n,
        "yes_trades": int(np.sum(side_yes)),
        "no_trades": int(np.sum(side_no)),
        "accuracy": accuracy,
        "yes_accuracy": yes_acc,
        "no_accuracy": no_acc,
        "total_pnl": float(np.sum(pnl)),
        "avg_pnl": float(np.mean(pnl)),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "loss_ratio": loss_ratio,
        "expected_value": float(ev),
        "trades_per_1000_rows": float((n / max(1, CURRENT_N_ROWS)) * 1000.0),
        "max_drawdown": max_drawdown_from_pnls(pnl),
        "sharpe_proxy": sharpe,
        "entry_0_00_0_50_pct": float(dist["0.00-0.50"]),
        "entry_0_50_0_65_pct": float(dist["0.50-0.65"]),
        "entry_0_65_0_75_pct": float(dist["0.65-0.75"]),
        "entry_0_75_0_85_pct": float(dist["0.75-0.85"]),
        "entry_0_85_1_00_pct": float(dist["0.85-1.00"]),
    }


def run_all_strategies(ctx: Context) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    all_rows: list[dict] = []
    trade_map: dict[str, pd.DataFrame] = {}

    # Strategy 1: Agreement YES
    cutoffs = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    windows_agree = [(60, 120), (60, 180), (60, 300), (30, 120), (30, 180)]
    max_entries_agree = [0.70, 0.75, 0.80, 0.85, 1.00]
    for cutoff, (min_s, max_s), max_e in product(cutoffs, windows_agree, max_entries_agree):
        fire = (
            (ctx.p_market >= cutoff)
            & (ctx.p_raw >= cutoff)
            & (ctx.seconds >= min_s)
            & (ctx.seconds <= max_s)
            & (ctx.p_market <= max_e)
        )
        tag = f"Agreement YES|cutoff={cutoff:.2f},window={min_s}-{max_s},max_entry={max_e:.2f}"
        tdf = build_trade_df(ctx, fire, np.zeros_like(fire, dtype=bool), tag)
        all_rows.append(metrics_from_trades("Agreement YES", tag.split("|", 1)[1], tdf))
        trade_map[tag] = tdf

    # Strategy 2: Agreement NO
    for cutoff, (min_s, max_s), max_e in product(cutoffs, windows_agree, max_entries_agree):
        no_cut = 1.0 - cutoff
        no_price = 1.0 - ctx.p_market
        fire = (
            (ctx.p_market <= no_cut)
            & (ctx.p_raw <= no_cut)
            & (ctx.seconds >= min_s)
            & (ctx.seconds <= max_s)
            & (no_price <= max_e)
        )
        tag = f"Agreement NO|cutoff={cutoff:.2f},window={min_s}-{max_s},max_entry={max_e:.2f}"
        tdf = build_trade_df(ctx, np.zeros_like(fire, dtype=bool), fire, tag)
        all_rows.append(metrics_from_trades("Agreement NO", tag.split("|", 1)[1], tdf))
        trade_map[tag] = tdf

    # Strategy 3: Mispricing
    thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
    windows_mp = [(60, 120), (60, 180), (60, 300), (30, 180), (30, 300)]
    max_entries_mp = [0.70, 0.75, 0.85, 1.00]
    directions = ["both", "bullish_only", "bearish_only"]
    for th, (min_s, max_s), max_e, direction in product(thresholds, windows_mp, max_entries_mp, directions):
        in_window = (ctx.seconds >= min_s) & (ctx.seconds <= max_s)
        bull = in_window & (ctx.p_raw > (ctx.p_market + th)) & (ctx.p_market <= max_e)
        no_price = 1.0 - ctx.p_market
        bear = in_window & (ctx.p_raw < (ctx.p_market - th)) & (no_price <= max_e)
        if direction == "bullish_only":
            bear = np.zeros_like(bear, dtype=bool)
        elif direction == "bearish_only":
            bull = np.zeros_like(bull, dtype=bool)
        tag = (
            f"Mispricing|threshold={th:.2f},window={min_s}-{max_s},max_entry={max_e:.2f},"
            f"direction={direction}"
        )
        tdf = build_trade_df(ctx, bull, bear, tag)
        all_rows.append(metrics_from_trades("Mispricing", tag.split("|", 1)[1], tdf))
        trade_map[tag] = tdf

    # Strategy 4: Dynamic cutoff (YES only)
    dynamic_floors = [0.55, 0.60, 0.65, 0.70]
    max_entries_dyn = [0.65, 0.70, 0.75, 0.80]
    windows_dyn = [(60, 180), (60, 300)]
    for floor, max_e, (min_s, max_s) in product(dynamic_floors, max_entries_dyn, windows_dyn):
        fire = (
            (ctx.p_market >= floor)
            & (ctx.p_raw >= floor)
            & (ctx.seconds >= min_s)
            & (ctx.seconds <= max_s)
            & (ctx.p_market <= max_e)
        )
        tag = f"Dynamic Cutoff|floor={floor:.2f},window={min_s}-{max_s},max_entry={max_e:.2f}"
        tdf = build_trade_df(ctx, fire, np.zeros_like(fire, dtype=bool), tag)
        all_rows.append(metrics_from_trades("Dynamic Cutoff", tag.split("|", 1)[1], tdf))
        trade_map[tag] = tdf

    # Strategy 5: Early entry agreement (YES only)
    early_fire = (
        (ctx.seconds >= 300)
        & (ctx.seconds <= 600)
        & (ctx.p_market >= 0.80)
        & (ctx.p_raw >= 0.80)
        & (ctx.p_market <= 0.80)
    )
    normal_fire = (
        (ctx.seconds >= 60)
        & (ctx.seconds <= 180)
        & (ctx.p_market >= 0.65)
        & (ctx.p_raw >= 0.65)
        & (ctx.p_market <= 0.80)
    )
    fire = early_fire | normal_fire
    tag = "Early Entry Agreement|early=300-600@0.80,normal=60-180@0.65,max_entry=0.80"
    tdf = build_trade_df(ctx, fire, np.zeros_like(fire, dtype=bool), tag)
    all_rows.append(metrics_from_trades("Early Entry Agreement", tag.split("|", 1)[1], tdf))
    trade_map[tag] = tdf

    # Strategy 6: Ensemble vote (YES only)
    fire = (
        (ctx.seconds >= 60)
        & (ctx.seconds <= 180)
        & (
            ((ctx.p_market >= 0.65) & (ctx.p_raw >= 0.65))
            | (ctx.p_raw > (ctx.p_market + 0.20))
        )
        & (ctx.p_market <= 0.80)
    )
    tag = "Ensemble Vote|window=60-180,entry<=0.80,agree>=0.65 OR mispricing_bullish>=0.20"
    tdf = build_trade_df(ctx, fire, np.zeros_like(fire, dtype=bool), tag)
    all_rows.append(metrics_from_trades("Ensemble Vote", tag.split("|", 1)[1], tdf))
    trade_map[tag] = tdf

    results = pd.DataFrame(all_rows)
    results = results.sort_values(["expected_value", "total_pnl"], ascending=[False, False]).reset_index(drop=True)
    return results, trade_map


def print_master_top20(results: pd.DataFrame) -> None:
    print("\nTOP 20 CONFIGS ACROSS ALL STRATEGIES")
    print("-" * 165)
    print(
        f"{'Strategy':<24} {'Params':<56} {'Trades':<7} {'Acc':<8} {'AvgPnL':<9} "
        f"{'TotalPnL':<10} {'EV':<9} {'LossRatio':<10}"
    )
    print("-" * 165)
    for row in results.head(20).itertuples(index=False):
        loss = "inf" if math.isinf(float(row.loss_ratio)) else f"{float(row.loss_ratio):.2f}"
        print(
            f"{str(row.strategy):<24} {str(row.params)[:56]:<56} {int(row.total_trades):<7} "
            f"{row.accuracy * 100:>6.2f}% {row.avg_pnl:>+8.3f} {row.total_pnl:>+10.2f} "
            f"{row.expected_value:>+8.3f} {loss:<10}"
        )
    print("-" * 165)


def print_per_strategy_winners(results: pd.DataFrame) -> pd.DataFrame:
    winners = (
        results.sort_values(["strategy", "expected_value", "total_pnl"], ascending=[True, False, False])
        .groupby("strategy", as_index=False)
        .first()
    )
    print("\nPER-STRATEGY WINNERS")
    print("-" * 155)
    print(f"{'Strategy':<24} {'Params':<56} {'Trades':<7} {'Acc':<8} {'AvgPnL':<9} {'TotalPnL':<10} {'EV':<9}")
    print("-" * 155)
    for row in winners.itertuples(index=False):
        print(
            f"{str(row.strategy):<24} {str(row.params)[:56]:<56} {int(row.total_trades):<7} "
            f"{row.accuracy * 100:>6.2f}% {row.avg_pnl:>+8.3f} {row.total_pnl:>+10.2f} {row.expected_value:>+8.3f}"
        )
    print("-" * 155)
    return winners


def print_realistic_daily_expectations(results: pd.DataFrame, n_test_rows: int, top_n: int = 10) -> None:
    rows_per_day = 96.0
    denom = n_test_rows / rows_per_day if n_test_rows > 0 else 1.0
    top = results.head(top_n).copy()
    top["trades_per_day"] = top["total_trades"] / max(denom, 1e-9)
    top["daily_pnl"] = top["trades_per_day"] * top["avg_pnl"]
    print("\nREALISTIC DAILY EXPECTATIONS (Top configs)")
    print("-" * 150)
    print(f"{'Strategy':<24} {'Trades/day':<10} {'AvgPnL/trade':<13} {'DailyPnL':<11} {'Config':<70}")
    print("-" * 150)
    for row in top.itertuples(index=False):
        print(
            f"{str(row.strategy):<24} {row.trades_per_day:<10.2f} {row.avg_pnl:<+13.3f} "
            f"{row.daily_pnl:<+11.2f} {str(row.params)[:70]:<70}"
        )
    print("-" * 150)


def print_entry_distribution(winners: pd.DataFrame) -> None:
    print("\nENTRY PRICE DISTRIBUTION (Per strategy winner)")
    print("-" * 125)
    print(
        f"{'Strategy':<24} {'0.00-0.50':>10} {'0.50-0.65':>10} {'0.65-0.75':>10} "
        f"{'0.75-0.85':>10} {'0.85-1.00':>10}"
    )
    print("-" * 125)
    for row in winners.itertuples(index=False):
        print(
            f"{str(row.strategy):<24} "
            f"{row.entry_0_00_0_50_pct*100:>9.1f}% "
            f"{row.entry_0_50_0_65_pct*100:>9.1f}% "
            f"{row.entry_0_65_0_75_pct*100:>9.1f}% "
            f"{row.entry_0_75_0_85_pct*100:>9.1f}% "
            f"{row.entry_0_85_1_00_pct*100:>9.1f}%"
        )
    print("-" * 125)


def print_price_drift_analysis(trades_df: pd.DataFrame, title: str) -> None:
    if trades_df.empty:
        print(f"\nPRICE DRIFT ANALYSIS ({title})\nNo trades.")
        return
    b = pd.cut(
        trades_df["entry_price"],
        bins=ENTRY_BUCKETS,
        labels=ENTRY_BUCKET_LABELS,
        right=False,
        include_lowest=True,
    )
    grp = (
        trades_df.assign(entry_bucket=b)
        .groupby("entry_bucket", observed=False)
        .agg(count=("upside", "count"), avg_upside=("upside", "mean"), avg_entry=("entry_price", "mean"))
        .reindex(ENTRY_BUCKET_LABELS, fill_value=0)
    )
    print(f"\nPRICE DRIFT ANALYSIS ({title})")
    print(grp.to_string(float_format=lambda x: f"{x:.4f}"))


def monte_carlo_significance(best_trades: pd.DataFrame, ctx: Context, iters: int = 1000, seed: int = 42) -> tuple[float, float]:
    if best_trades.empty:
        return 1.0, 0.0
    rng = np.random.default_rng(seed)
    trade_idx = best_trades["row_idx"].to_numpy(dtype=int)
    side_yes = (best_trades["side"].to_numpy() == "YES")
    p_market = best_trades["p_market"].to_numpy(dtype=float)
    entry_yes = best_trades["entry_price"].to_numpy(dtype=float)
    entry_no = 1.0 - p_market
    contracts = best_trades["contracts"].to_numpy(dtype=float)
    actual_total = float(np.sum(best_trades["pnl"].to_numpy(dtype=float)))

    shuffled_beats = 0
    outcomes = ctx.outcome_yes.copy()
    for _ in range(iters):
        shuf = rng.permutation(outcomes)
        trade_out = shuf[trade_idx]
        yes_win = side_yes & (trade_out == 1)
        no_win = (~side_yes) & (trade_out == 0)
        gross_yes = np.where(yes_win, (1.0 - p_market) * contracts, -entry_yes * contracts)
        gross_no = np.where(no_win, p_market * contracts, -entry_no * contracts)
        gross_pnl = np.where(side_yes, gross_yes, gross_no)
        spread_total = SPREAD_COST * contracts
        fee = np.where((side_yes & yes_win) | (~side_yes & no_win),
                       FEE_RATE * np.maximum(gross_pnl, 0.0), 0.0)
        pnl = gross_pnl - spread_total - fee
        if float(np.sum(pnl)) >= actual_total:
            shuffled_beats += 1
    frac = shuffled_beats / iters
    return frac, actual_total


def walk_forward_backtest(
    df: pd.DataFrame,
    n_folds: int = 5,
) -> list[dict]:
    """Walk-forward backtest: for each fold, train on all prior data, evaluate on the fold.

    Returns a list of dicts, one per fold, with fold index, train size, test size,
    and the full strategy results DataFrame for that fold.
    """
    df_sorted = df.sort_values(by=["close_ts", "market_ticker", "entry_bucket"]).reset_index(drop=True)
    fold_size = len(df_sorted) // n_folds
    remainder = len(df_sorted) % n_folds

    fold_results: list[dict] = []
    boundaries = []
    start = 0
    for i in range(n_folds):
        size = fold_size + (1 if i < remainder else 0)
        boundaries.append((start, start + size))
        start += size

    model, model_features = load_model_bundle(MODEL_PATH)
    missing = sorted(set(model_features) - set(df_sorted.columns))
    if missing:
        raise ValueError(f"Missing model features in dataset: {missing}")

    for fold_idx in range(n_folds):
        test_start, test_end = boundaries[fold_idx]
        # Train on all data before this fold
        train_df = df_sorted.iloc[:test_start].copy()
        test_df = df_sorted.iloc[test_start:test_end].copy()

        if len(train_df) == 0 or len(test_df) == 0:
            fold_results.append({
                "fold": fold_idx,
                "train_size": len(train_df),
                "test_size": len(test_df),
                "results": pd.DataFrame(),
            })
            continue

        # Fit model predictions on training data + test data
        # Re-fit is not needed for sklearn models that are already trained;
        # we use predict_proba from the pre-trained model
        train_df["p_raw"] = model.predict_proba(train_df[model_features])[:, 1]
        test_df["p_raw"] = model.predict_proba(test_df[model_features])[:, 1]

        test_ctx = Context(
            p_market=test_df["price_now"].to_numpy(dtype=float),
            p_raw=test_df["p_raw"].to_numpy(dtype=float),
            seconds=test_df["seconds_to_close"].to_numpy(dtype=int),
            outcome_yes=test_df[TARGET].to_numpy(dtype=int),
            row_index=np.arange(len(test_df), dtype=int),
            n_rows=len(test_df),
        )

        global CURRENT_N_ROWS
        old_n = CURRENT_N_ROWS
        CURRENT_N_ROWS = len(test_df)
        results, _ = run_all_strategies(test_ctx)
        CURRENT_N_ROWS = old_n

        fold_results.append({
            "fold": fold_idx,
            "train_size": len(train_df),
            "test_size": len(test_df),
            "results": results,
        })

    return fold_results


def regime_analysis(
    test_df: pd.DataFrame,
    trades_df: pd.DataFrame,
) -> dict[str, dict]:
    """Split trade results by volatility regime (high vs low, based on median volatility_5m).

    Parameters
    ----------
    test_df : DataFrame with a ``volatility_5m`` column, indexed or aligned to
        the same row order used to build *trades_df*.
    trades_df : DataFrame of trades produced by :func:`build_trade_df`.

    Returns
    -------
    dict with keys ``"low_vol"`` and ``"high_vol"``, each mapping to a metrics
    dict produced by :func:`metrics_from_trades`.
    """
    if trades_df.empty or "volatility_5m" not in test_df.columns:
        empty = metrics_from_trades("regime", "", pd.DataFrame())
        return {"low_vol": empty, "high_vol": empty}

    vol_median = test_df["volatility_5m"].median()
    # Map volatility_5m from test_df onto each trade via row_idx
    vol_series = test_df["volatility_5m"].reset_index(drop=True)
    trades_with_vol = trades_df.copy()
    trades_with_vol["volatility_5m"] = trades_with_vol["row_idx"].map(
        lambda idx: vol_series.iloc[idx] if idx < len(vol_series) else float("nan")
    )

    low_vol_trades = trades_with_vol[trades_with_vol["volatility_5m"] <= vol_median]
    high_vol_trades = trades_with_vol[trades_with_vol["volatility_5m"] > vol_median]

    return {
        "low_vol": metrics_from_trades("regime", f"low_vol (vol<={vol_median:.6f})", low_vol_trades),
        "high_vol": metrics_from_trades("regime", f"high_vol (vol>{vol_median:.6f})", high_vol_trades),
        "vol_median": vol_median,
    }


def print_honest_conclusion(results: pd.DataFrame) -> None:
    eligible = results[(results["total_trades"] >= 50) & (results["expected_value"] > 0)]
    print("\nHONEST CONCLUSION:")
    if not eligible.empty:
        best = eligible.iloc[0]
        print(
            "Positive-EV strategy with 50+ trades found: "
            f"{best['strategy']} | {best['params']} | trades={int(best['total_trades'])} | "
            f"EV={best['expected_value']:+.3f} | TotalPnL={best['total_pnl']:+.2f}"
        )
    else:
        print("No strategy showed consistent positive EV with 50+ trades on this dataset. Possible explanations:")
        print("1. Historical dataset may have different market regime")
        print("2. All edges may be eliminated by spreads/fees in practice")
        print("3. Larger dataset needed for statistical significance")
        print("4. Model needs retraining with updated data")


CURRENT_N_ROWS = 0


def main() -> None:
    global CURRENT_N_ROWS
    print("Loading dataset...")
    df = load_dataset(DATA_PATH)
    test_df = temporal_test_split(df)
    CURRENT_N_ROWS = len(test_df)
    print(f"Total rows: {len(df)} | Test rows: {len(test_df)}")

    print("Loading model and computing p_raw for all test rows...")
    model, model_features = load_model_bundle(MODEL_PATH)
    missing = sorted(set(model_features) - set(test_df.columns))
    if missing:
        raise ValueError(f"Missing model features in test set: {missing}")
    if model_features != RAW_FEATURES:
        print("Note: using feature list embedded in model artifact.")
    test_df = test_df.copy()
    test_df["p_raw"] = model.predict_proba(test_df[model_features])[:, 1]

    ctx = Context(
        p_market=test_df["price_now"].to_numpy(dtype=float),
        p_raw=test_df["p_raw"].to_numpy(dtype=float),
        seconds=test_df["seconds_to_close"].to_numpy(dtype=int),
        outcome_yes=test_df[TARGET].to_numpy(dtype=int),
        row_index=np.arange(len(test_df), dtype=int),
        n_rows=len(test_df),
    )

    print("Running full strategy grid...")
    results, trade_map = run_all_strategies(ctx)
    print(f"Total configs tested: {len(results)}")

    print_master_top20(results)
    winners = print_per_strategy_winners(results)
    print_realistic_daily_expectations(results, len(test_df), top_n=10)
    print_entry_distribution(winners)

    # Price drift analysis on top overall config + strategy winners
    best_overall = results.iloc[0]
    best_key = f"{best_overall['strategy']}|{best_overall['params']}"
    print_price_drift_analysis(trade_map.get(best_key, pd.DataFrame()), "Best overall config")
    for row in winners.itertuples(index=False):
        key = f"{row.strategy}|{row.params}"
        print_price_drift_analysis(trade_map.get(key, pd.DataFrame()), f"{row.strategy} winner")

    # Save outputs
    results.to_csv(RESULTS_CSV, index=False)
    top_configs = results.head(20).copy()
    top_trade_frames: list[pd.DataFrame] = []
    for row in top_configs.itertuples(index=False):
        key = f"{row.strategy}|{row.params}"
        tdf = trade_map.get(key, pd.DataFrame()).copy()
        if tdf.empty:
            continue
        tdf["strategy"] = row.strategy
        tdf["params"] = row.params
        top_trade_frames.append(tdf)
    top_trades = pd.concat(top_trade_frames, ignore_index=True) if top_trade_frames else pd.DataFrame()
    top_trades.to_csv(TOP_TRADES_CSV, index=False)
    print(f"\nSaved full results to {RESULTS_CSV}")
    print(f"Saved top configs trades to {TOP_TRADES_CSV}")

    # Monte Carlo on best config
    print("\nMONTE CARLO SIGNIFICANCE TEST (best config)")
    best_trades = trade_map.get(best_key, pd.DataFrame())
    frac, actual_total = monte_carlo_significance(best_trades, ctx, iters=1000, seed=42)
    print(f"Best config: {best_overall['strategy']} | {best_overall['params']}")
    print(f"Actual total PnL: {actual_total:+.2f}")
    print(f"Fraction of random shuffles that beat actual PnL: {frac*100:.2f}%")
    if frac < 0.05:
        print("Strategy shows statistically significant edge")
    else:
        print("Strategy may be fitting to noise")

    print_honest_conclusion(results)


if __name__ == "__main__":
    main()
