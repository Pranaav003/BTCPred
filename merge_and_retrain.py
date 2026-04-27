"""Merge historical + live training rows and retrain model artifact."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live-only", action="store_true", help="Train on live data only, ignore historical")
    parser.add_argument("--min-live-rows", type=int, default=100, help="Minimum live rows required to proceed")
    parser.add_argument("--live-weight", type=float, default=3.0, help="How much to upweight recent live rows vs historical")
    args = parser.parse_args()

    try:
        live_df = pd.read_csv(LIVE_PATH)
        print(f"Live rows loaded: {len(live_df)}")
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

    missing_features = sorted(set(RAW_FEATURES + [TARGET]) - set(df.columns))
    if missing_features:
        raise ValueError(f"Missing required columns: {missing_features}")

    split_idx = int(len(df) * (1 - TEST_SIZE))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    x_train = train_df[RAW_FEATURES]
    y_train = train_df[TARGET].astype(int)
    x_test = test_df[RAW_FEATURES]
    y_test = test_df[TARGET].astype(int)

    w_train = weights[:split_idx] if weights is not None else None

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
        MODEL_OUTPUT,
    )

    print(f"\nSaved to {MODEL_OUTPUT}")
    print("Restart Flask to load the new model.")


if __name__ == "__main__":
    main()
