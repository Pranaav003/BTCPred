"""Standalone mispricing strategy backtest over the historical 30k dataset."""

from __future__ import annotations

import math
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

DATA_PATH = "kalshi_btc15m_dataset_30k.csv"
MODEL_PATH = "raw_feature_model.pkl"
TARGET = "final_outcome_yes"
TEST_SIZE = 0.20  # only evaluate on test set (last 20%)

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

MISPRICING_THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30]
TIME_WINDOWS = [
    (60, 120),  # conservative
    (60, 180),  # moderate
    (60, 300),  # aggressive
]
MAX_ENTRY_PRICES = [0.75, 0.85, 1.00]  # YES entry filter
BASE_TRADE_SIZE = 20.0
FEE_RATE = 0.01  # 1% on profits (matching Kalshi)
SPREAD_COST = 0.02  # 2 cents per contract

RESULTS_OUTPUT = "backtest_results.csv"
BEST_TRADES_OUTPUT = "backtest_best_trades.csv"


def load_dataset(path: str) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    df = pd.read_csv(csv_path, dtype={"series_ticker": str})
    required = set(RAW_FEATURES + [TARGET, "price_now", "seconds_to_close", "entry_bucket", "close_ts"])
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df["avg_trade_price_1m"] = df["avg_trade_price_1m"].fillna(df["price_now"])
    df["avg_trade_price_3m"] = df["avg_trade_price_3m"].fillna(df["price_now"])
    df = df.sort_values(by=["close_ts", "market_ticker", "entry_bucket"]).reset_index(drop=True)
    return df


def load_model_bundle(path: str) -> tuple[object, list[str]]:
    obj = joblib.load(path)
    if isinstance(obj, dict):
        model = obj.get("model")
        features = obj.get("features")
        if model is None:
            raise ValueError("MODEL_PATH dict does not include key 'model'.")
        if not isinstance(features, list) or not features:
            features = RAW_FEATURES
        return model, list(features)
    return obj, RAW_FEATURES


def temporal_test_split(df: pd.DataFrame, test_size: float) -> pd.DataFrame:
    if not 0 < test_size < 1:
        raise ValueError("TEST_SIZE must be between 0 and 1.")
    split_idx = int(len(df) * (1 - test_size))
    if split_idx <= 0 or split_idx >= len(df):
        raise ValueError("Dataset too small for configured TEST_SIZE.")
    return df.iloc[split_idx:].copy().reset_index(drop=True)


def compute_summary(records: list[dict]) -> dict:
    total_trades = len(records)
    if total_trades == 0:
        return {
            "trades": 0,
            "wins": 0,
            "accuracy": 0.0,
            "total_pnl": 0.0,
            "avg_pnl": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "loss_ratio": float("inf"),
            "yes_count": 0,
            "yes_accuracy": 0.0,
            "yes_total_pnl": 0.0,
            "no_count": 0,
            "no_accuracy": 0.0,
            "no_total_pnl": 0.0,
            "gap_10_20_count": 0,
            "gap_10_20_pnl": 0.0,
            "gap_20_30_count": 0,
            "gap_20_30_pnl": 0.0,
            "gap_30_plus_count": 0,
            "gap_30_plus_pnl": 0.0,
        }

    rec_df = pd.DataFrame(records)
    wins = int(rec_df["won"].sum())
    total_pnl = float(rec_df["pnl"].sum())
    avg_pnl = float(rec_df["pnl"].mean())
    pos = rec_df[rec_df["pnl"] > 0]["pnl"]
    neg = rec_df[rec_df["pnl"] < 0]["pnl"]
    avg_win = float(pos.mean()) if not pos.empty else 0.0
    avg_loss = float(neg.mean()) if not neg.empty else 0.0
    loss_ratio = abs(avg_loss) / avg_win if avg_win > 0 else float("inf")

    yes_df = rec_df[rec_df["side"] == "YES"]
    no_df = rec_df[rec_df["side"] == "NO"]

    gap_10_20 = rec_df[(rec_df["gap"] >= 0.10) & (rec_df["gap"] < 0.20)]
    gap_20_30 = rec_df[(rec_df["gap"] >= 0.20) & (rec_df["gap"] < 0.30)]
    gap_30_plus = rec_df[rec_df["gap"] >= 0.30]

    return {
        "trades": total_trades,
        "wins": wins,
        "accuracy": wins / total_trades,
        "total_pnl": total_pnl,
        "avg_pnl": avg_pnl,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "loss_ratio": loss_ratio,
        "yes_count": int(len(yes_df)),
        "yes_accuracy": float(yes_df["won"].mean()) if not yes_df.empty else 0.0,
        "yes_total_pnl": float(yes_df["pnl"].sum()) if not yes_df.empty else 0.0,
        "no_count": int(len(no_df)),
        "no_accuracy": float(no_df["won"].mean()) if not no_df.empty else 0.0,
        "no_total_pnl": float(no_df["pnl"].sum()) if not no_df.empty else 0.0,
        "gap_10_20_count": int(len(gap_10_20)),
        "gap_10_20_pnl": float(gap_10_20["pnl"].sum()) if not gap_10_20.empty else 0.0,
        "gap_20_30_count": int(len(gap_20_30)),
        "gap_20_30_pnl": float(gap_20_30["pnl"].sum()) if not gap_20_30.empty else 0.0,
        "gap_30_plus_count": int(len(gap_30_plus)),
        "gap_30_plus_pnl": float(gap_30_plus["pnl"].sum()) if not gap_30_plus.empty else 0.0,
    }


def simulate_mispricing_strategy(
    test_df: pd.DataFrame,
    threshold: float,
    min_seconds: int,
    max_seconds: int,
    max_entry: float,
) -> tuple[list[dict], dict]:
    records: list[dict] = []
    for row in test_df.itertuples(index=False):
        p_market = float(row.price_now)
        p_raw = float(row.p_raw_pred)
        seconds = int(row.seconds_to_close)
        outcome_yes = int(getattr(row, TARGET))
        gap = abs(p_raw - p_market)

        if not (min_seconds <= seconds <= max_seconds):
            continue

        if p_raw > p_market + threshold:
            side = "YES"
            entry_price = p_market
        elif p_raw < p_market - threshold:
            side = "NO"
            entry_price = 1.0 - p_market
        else:
            continue

        if entry_price <= 0:
            continue

        if side == "YES" and p_market > max_entry:
            continue
        if side == "NO" and (1.0 - p_market) > max_entry:
            continue

        contracts = BASE_TRADE_SIZE / entry_price

        if side == "YES":
            won = outcome_yes == 1
            gross_pnl = (1.0 - p_market) * contracts if won else -entry_price * contracts
        else:
            won = outcome_yes == 0
            gross_pnl = p_market * contracts if won else -(1.0 - p_market) * contracts

        spread_total = SPREAD_COST * contracts
        fee = FEE_RATE * max(gross_pnl, 0.0) if won else 0.0
        pnl = gross_pnl - spread_total - fee

        records.append(
            {
                "side": side,
                "entry_price": entry_price,
                "contracts": contracts,
                "gap": gap,
                "pnl": float(pnl),
                "won": bool(won),
                "seconds": seconds,
                "p_market": p_market,
                "p_raw": p_raw,
                "outcome_yes": outcome_yes,
                "entry_bucket": int(row.entry_bucket),
                "close_ts": int(row.close_ts),
            }
        )

    return records, compute_summary(records)


def simulate_agreement_yes(
    test_df: pd.DataFrame,
    min_seconds: int,
    max_seconds: int,
    yes_cutoff: float = 0.65,
) -> dict:
    records: list[dict] = []
    for row in test_df.itertuples(index=False):
        p_market = float(row.price_now)
        p_raw = float(row.p_raw_pred)
        seconds = int(row.seconds_to_close)
        outcome_yes = int(getattr(row, TARGET))
        if not (min_seconds <= seconds <= max_seconds):
            continue
        if not (p_market >= yes_cutoff and p_raw >= yes_cutoff):
            continue
        if p_market <= 0:
            continue

        contracts = BASE_TRADE_SIZE / p_market
        won = outcome_yes == 1
        gross_pnl = (1.0 - p_market) * contracts if won else -p_market * contracts
        spread_total = SPREAD_COST * contracts
        fee = FEE_RATE * max(gross_pnl, 0.0) if won else 0.0
        pnl = gross_pnl - spread_total - fee
        records.append({"pnl": float(pnl), "won": bool(won)})

    if not records:
        return {"trades": 0, "accuracy": 0.0, "total_pnl": 0.0}
    rdf = pd.DataFrame(records)
    return {
        "trades": int(len(rdf)),
        "accuracy": float(rdf["won"].mean()),
        "total_pnl": float(rdf["pnl"].sum()),
    }


def print_master_table(results_df: pd.DataFrame) -> None:
    if results_df.empty:
        print("\nNo results to display.")
        return

    display_df = results_df.copy()
    max_total = float(display_df["total_pnl"].max())
    max_acc = float(display_df["accuracy"].max())
    max_avg = float(display_df["avg_pnl"].max())
    max_trades = int(display_df["trades"].max())
    min_loss_ratio = float(display_df.replace({"loss_ratio": {np.inf: np.nan}})["loss_ratio"].min(skipna=True))

    print("\nMASTER RESULTS TABLE (sorted by TotalPnL desc)")
    print("-" * 126)
    print(
        f"{'Threshold':<10} {'Window':<9} {'MaxEntry':<8} {'Trades':<7} {'YES':<5} {'NO':<5} "
        f"{'Accuracy':<10} {'AvgPnL':<10} {'TotalPnL':<11} {'LossRatio':<10}"
    )
    print("-" * 126)
    for row in display_df.itertuples(index=False):
        loss_ratio = float(row.loss_ratio)
        loss_str = "inf" if math.isinf(loss_ratio) else f"{loss_ratio:.2f}"
        if math.isfinite(loss_ratio) and math.isfinite(min_loss_ratio) and abs(loss_ratio - min_loss_ratio) < 1e-12:
            loss_str += "*"
        trades = f"{int(row.trades)}" + ("*" if int(row.trades) == max_trades else "")
        acc = f"{row.accuracy * 100:.2f}%" + ("*" if abs(float(row.accuracy) - max_acc) < 1e-12 else "")
        avg_pnl = f"{row.avg_pnl:+.3f}" + ("*" if abs(float(row.avg_pnl) - max_avg) < 1e-12 else "")
        total_pnl = f"{row.total_pnl:+.2f}" + ("*" if abs(float(row.total_pnl) - max_total) < 1e-12 else "")
        print(
            f"{row.threshold:<10.2f} {str(row.window_label):<9} {row.max_entry:<8.2f} {trades:<7} "
            f"{int(row.yes_count):<5} {int(row.no_count):<5} {acc:<10} {avg_pnl:<10} {total_pnl:<11} {loss_str:<10}"
        )
    print("-" * 126)
    print("* = best value in that column")


def print_top_5_details(results_df: pd.DataFrame) -> None:
    top5 = results_df.head(5)
    print("\nTOP 5 PARAMETER COMBINATIONS BY TOTAL P/L")
    for i, row in enumerate(top5.itertuples(index=False), start=1):
        print(
            f"{i}. threshold={row.threshold:.2f}, window={row.window_label}, max_entry={row.max_entry:.2f} | "
            f"trades={int(row.trades)}, acc={row.accuracy * 100:.2f}%, "
            f"total_pnl={row.total_pnl:+.2f}, avg_pnl={row.avg_pnl:+.3f}, "
            f"YES={int(row.yes_count)} ({row.yes_accuracy * 100:.1f}%, {row.yes_total_pnl:+.2f}), "
            f"NO={int(row.no_count)} ({row.no_accuracy * 100:.1f}%, {row.no_total_pnl:+.2f})"
        )


def print_best_breakdown(best_row: pd.Series, best_trades_df: pd.DataFrame) -> None:
    print("\nBEST PARAMETER COMBINATION DETAILED BREAKDOWN")
    print(
        f"threshold={best_row['threshold']:.2f}, window={best_row['window_label']}, "
        f"max_entry={best_row['max_entry']:.2f}, trades={int(best_row['trades'])}, "
        f"total_pnl={best_row['total_pnl']:+.2f}, accuracy={best_row['accuracy'] * 100:.2f}%"
    )

    if best_trades_df.empty:
        print("No trades for best combination.")
        return

    bt = best_trades_df.copy()
    bt["month"] = pd.to_datetime(bt["close_ts"], unit="s", utc=True).dt.tz_convert(None).dt.to_period("M").astype(str)
    bt["gap_bucket"] = np.select(
        [bt["gap"] < 0.20, bt["gap"] < 0.30],
        ["0.10-0.20", "0.20-0.30"],
        default="0.30+",
    )

    print("\nMonthly PnL")
    print(bt.groupby("month")["pnl"].sum().sort_index().to_string())

    print("\nPnL by entry_bucket")
    print(bt.groupby("entry_bucket")["pnl"].agg(["count", "sum", "mean"]).sort_index().to_string())

    print("\nPnL by gap bucket")
    print(bt.groupby("gap_bucket")["pnl"].agg(["count", "sum", "mean"]).to_string())

    print("\nDistribution of entry prices")
    print(bt["entry_price"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).to_string())

    cols = ["side", "entry_price", "contracts", "gap", "pnl", "won", "seconds", "p_market", "p_raw", "outcome_yes"]
    print("\n10 Best Individual Trades")
    print(bt.sort_values("pnl", ascending=False).head(10)[cols].to_string(index=False))
    print("\n10 Worst Individual Trades")
    print(bt.sort_values("pnl", ascending=True).head(10)[cols].to_string(index=False))


def print_agreement_comparison(results_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    print("\nAGREEMENT (YES @ 0.65) VS MISPRICING COMPARISON")
    print(f"{'Window':<10} {'AgreementPnL':<14} {'AgreementTrades':<16} {'BestMispricingPnL':<18}")
    for window in TIME_WINDOWS:
        min_s, max_s = window
        label = f"{min_s}-{max_s}"
        agreement = simulate_agreement_yes(test_df, min_s, max_s, yes_cutoff=0.65)
        mispricing_best = float(results_df[results_df["window_label"] == label]["total_pnl"].max())
        print(
            f"{label:<10} {agreement['total_pnl']:+.2f}{'':<6} "
            f"{int(agreement['trades']):<16} {mispricing_best:+.2f}"
        )


def main() -> None:
    print("Loading dataset and model...")
    df = load_dataset(DATA_PATH)
    test_df = temporal_test_split(df, TEST_SIZE)

    model, model_features = load_model_bundle(MODEL_PATH)
    missing_model_features = sorted(set(model_features) - set(test_df.columns))
    if missing_model_features:
        raise ValueError(f"Test set missing model features: {missing_model_features}")

    if model_features != RAW_FEATURES:
        print("Note: using feature list embedded in model artifact.")

    X_test = test_df[model_features]
    test_df["p_raw_pred"] = model.predict_proba(X_test)[:, 1]

    results: list[dict] = []
    combo_records: dict[tuple[float, int, int, float], list[dict]] = {}

    for threshold, (min_s, max_s), max_entry in product(
        MISPRICING_THRESHOLDS,
        TIME_WINDOWS,
        MAX_ENTRY_PRICES,
    ):
        records, summary = simulate_mispricing_strategy(
            test_df=test_df,
            threshold=threshold,
            min_seconds=min_s,
            max_seconds=max_s,
            max_entry=max_entry,
        )
        key = (threshold, min_s, max_s, max_entry)
        combo_records[key] = records
        results.append(
            {
                "threshold": threshold,
                "window_min_seconds": min_s,
                "window_max_seconds": max_s,
                "window_label": f"{min_s}-{max_s}",
                "max_entry": max_entry,
                **summary,
            }
        )

    results_df = pd.DataFrame(results).sort_values(by="total_pnl", ascending=False).reset_index(drop=True)
    print_master_table(results_df)
    print_top_5_details(results_df)

    best = results_df.iloc[0]
    best_key = (
        float(best["threshold"]),
        int(best["window_min_seconds"]),
        int(best["window_max_seconds"]),
        float(best["max_entry"]),
    )
    best_trades_df = pd.DataFrame(combo_records[best_key]).sort_values(by="close_ts").reset_index(drop=True)
    print_best_breakdown(best, best_trades_df)
    print_agreement_comparison(results_df, test_df)

    results_df.to_csv(RESULTS_OUTPUT, index=False)
    best_trades_df.to_csv(BEST_TRADES_OUTPUT, index=False)
    print(f"\nSaved full results to {RESULTS_OUTPUT}")
    print(f"Saved best combo trades to {BEST_TRADES_OUTPUT}")

    # Recommendation block
    if not best_trades_df.empty:
        ts_min = float(best_trades_df["close_ts"].min())
        ts_max = float(best_trades_df["close_ts"].max())
        span_days = max((ts_max - ts_min) / 86400.0, 1e-9)
        trades_per_day = float(len(best_trades_df) / span_days)
        daily_pnl = float(best["total_pnl"] / span_days)
    else:
        trades_per_day = 0.0
        daily_pnl = 0.0

    print("\nRECOMMENDATION:")
    print(f"Best threshold: {best['threshold']:.2f}")
    print(f"Best window: {int(best['window_min_seconds'])}s-{int(best['window_max_seconds'])}s")
    print(f"Best max entry: {best['max_entry']:.2f}")
    print(f"Expected accuracy: {best['accuracy'] * 100:.2f}%")
    print(f"Expected avg PnL per trade: {best['avg_pnl']:+.2f}")
    print(f"Expected trades per day (extrapolated): ~{trades_per_day:.2f}")
    print(f"Expected daily PnL: {daily_pnl:+.2f}")


if __name__ == "__main__":
    main()
