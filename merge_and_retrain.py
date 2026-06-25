"""Merge historical + live training rows and retrain model artifact."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline

HISTORICAL_PATH = "kalshi_btc15m_dataset_30k.csv"
LIVE_PATH = "live_training_data.csv"
MODEL_OUTPUT = "raw_feature_model.pkl"
TARGET = "final_outcome_yes"
TEST_SIZE = 0.20

RAW_FEATURES = [
    "seconds_to_close", "entry_bucket",
    "return_1m", "return_3m", "return_5m",
    "volatility_3m", "volatility_5m", "range_5m",
    "abs_return_1m", "trade_count_1m", "trade_count_3m",
    "trade_count_5m", "volume_1m", "volume_3m", "volume_5m",
    "avg_trade_price_1m", "avg_trade_price_3m",
    "momentum_1m", "momentum_3m", "momentum_5m",
    "momentum_acceleration", "price_velocity_5m",
    "flip_count_5m", "return_1m_x_inv_time",
    "return_3m_x_inv_time", "volatility_5m_x_inv_time",
]


def check_feature_drift(historical_df, live_df, features, alpha=0.05):
    """Run KS test on key features to detect distribution shift."""
    from scipy.stats import ks_2samp
    drift_details = {}
    drift_detected = False
    for feat in features:
        if feat not in historical_df.columns or feat not in live_df.columns:
            continue
        hist_vals = historical_df[feat].dropna().values
        live_vals = live_df[feat].dropna().values
        if len(hist_vals) < 10 or len(live_vals) < 10:
            continue
        stat, p_val = ks_2samp(hist_vals, live_vals)
        drift_details[feat] = (stat, p_val)
        if p_val < alpha:
            drift_detected = True
            print(f"  DRIFT: {feat}: KS stat={stat:.4f}, p={p_val:.4f}")
    return drift_detected, drift_details


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live-only", action="store_true", help="Train on live data only, ignore historical")
    parser.add_argument("--min-live-rows", type=int, default=100, help="Minimum live rows required to proceed")
    parser.add_argument("--live-weight", type=float, default=2.0, help="How much to upweight recent live rows vs historical")
    parser.add_argument(
        "--model-output",
        type=str,
        default=MODEL_OUTPUT,
        help="Path for joblib bundle (default: raw_feature_model.pkl in cwd)",
    )
    parser.add_argument(
        "--live-data",
        type=str,
        default=LIVE_PATH,
        help="Path to live training CSV (default: live_training_data.csv)",
    )
    args = parser.parse_args()

    live_path = args.live_data
    model_out = args.model_output

    try:
        live_df = pd.read_csv(live_path)
        print(f"Live rows loaded from {live_path}: {len(live_df)}")
    except FileNotFoundError:
        print("No live_training_data.csv found.")
        print("Export it first from the app: Analytics > Export > Download Live Training Data")
        print("Or use the API: GET /api/export/live-training-data")
        sys.exit(1)

    live_df["source"] = "live"

    if len(live_df) < args.min_live_rows:
        print(f"Only {len(live_df)} live rows. Minimum is {args.min_live_rows}.")
        print("Keep collecting data and retry later.")
        remaining = max(0, args.min_live_rows - len(live_df))
        print(f"Estimated time to minimum: ~{remaining * 15 / 60:.0f} minutes at 15s polling")
        sys.exit(1)

    hist_df = None
    if args.live_only:
        df = live_df
        print("Training on live data only")
    else:
        try:
            hist_df = pd.read_csv(HISTORICAL_PATH, dtype={"series_ticker": str})
            hist_df["source"] = "historical"
            print(f"Historical rows: {len(hist_df)}")
        except FileNotFoundError:
            print("Historical dataset not found, using live only")
            df = live_df
        else:
            df = pd.concat([hist_df, live_df], ignore_index=True)
            print(f"Combined rows: {len(df)}")

    for col in ["avg_trade_price_1m", "avg_trade_price_3m"]:
        if col in df.columns:
            df[col] = df[col].fillna(df["price_now"])

    if "close_ts" in df.columns:
        df = df.sort_values("close_ts").reset_index(drop=True)

    if not args.live_only and "source" in df.columns:
        weights = np.where(df["source"] == "live", args.live_weight, 1.0)
        print(f"Live data upweighted {args.live_weight}x")
    else:
        weights = None

    # --- Concept drift detection ---
    drift_features = ["return_1m", "return_3m", "volatility_3m", "volatility_5m", "trade_count_1m"]
    if hist_df is not None:
        print("\nChecking for feature drift (historical vs live)...")
        drift_detected, drift_details = check_feature_drift(hist_df, live_df, drift_features)
        if drift_detected:
            print("WARNING: Concept drift detected — live feature distributions differ from historical.")
            print("  The retrained model may not generalise well. Consider collecting more live data.")
        else:
            print("  No significant drift detected across key features.")

    missing_features = sorted(set(RAW_FEATURES + [TARGET]) - set(df.columns))
    if missing_features:
        raise ValueError(f"Missing required columns: {missing_features}")

    # Market-level split to prevent data leakage
    if "market_ticker" in df.columns:
        market_order = (
            df.groupby("market_ticker")["close_ts"]
            .min()
            .sort_values()
            .index
            .tolist()
        )
        n_test_markets = max(1, int(len(market_order) * TEST_SIZE))
        n_train_markets = len(market_order) - n_test_markets
        train_tickers = set(market_order[:n_train_markets])
        test_tickers = set(market_order[n_train_markets:])
        train_df = df[df["market_ticker"].isin(train_tickers)].copy()
        test_df = df[df["market_ticker"].isin(test_tickers)].copy()
        max_train_ts = train_df["close_ts"].max()
        min_test_ts = test_df["close_ts"].min()
        if min_test_ts - max_train_ts < 900:
            embargo_cutoff = max_train_ts + 900
            test_df = test_df[test_df["close_ts"] >= embargo_cutoff].copy()
    else:
        split_idx = int(len(df) * (1 - TEST_SIZE))
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]

    x_train = train_df[RAW_FEATURES]
    y_train = train_df[TARGET].astype(int)
    x_test = test_df[RAW_FEATURES]
    y_test = test_df[TARGET].astype(int)

    w_train = weights[:len(train_df)] if weights is not None else None

    print(f"\nTrain: {len(x_train)} rows | Test: {len(x_test)} rows")
    print(f"Train class balance: {y_train.mean():.1%} YES")

    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=8,
                    min_samples_leaf=10,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    print("Training...")
    if w_train is not None:
        model.fit(x_train, y_train, model__sample_weight=w_train)
    else:
        model.fit(x_train, y_train)

    p = model.predict_proba(x_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, p >= 0.5)),
        "roc_auc": float(roc_auc_score(y_test, p)),
        "brier": float(brier_score_loss(y_test, p)),
        "log_loss": float(log_loss(y_test, p)),
    }

    import sklearn
    import sys as _sys

    print("\nTest metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # --- Model comparison guard ---
    old_model_path = model_out
    try:
        old_bundle = joblib.load(old_model_path)
        old_model = old_bundle.get("model") or old_bundle.get("pipeline")
        if old_model is not None:
            old_preds = old_model.predict_proba(x_test)[:, 1]
            old_brier = brier_score_loss(y_test, old_preds)
            new_brier = metrics["brier"]
            print(f"Old model Brier: {old_brier:.4f}")
            print(f"New model Brier: {new_brier:.4f}")
            if new_brier > old_brier + 0.01:
                print("WARNING: New model is WORSE than old model. Saving anyway, but consider rolling back.")
    except Exception as exc:
        print(f"Could not compare with old model: {exc}")

    out_path = Path(model_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "features": RAW_FEATURES,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "test_metrics": metrics,
            "n_train": len(x_train),
            "n_test": len(x_test),
            "model_type": "RandomForest",
            "sklearn_version": sklearn.__version__,
            "python_version": _sys.version,
            "live_rows": len(live_df),
            "historical_rows": len(df) - len(live_df),
            "live_weight": args.live_weight if weights is not None else 1.0,
        },
        model_out,
    )

    print(f"\nSaved to {model_out}")
    print("Restart Flask to load the new model.")


if __name__ == "__main__":
    main()
