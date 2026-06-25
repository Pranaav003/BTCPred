"""Train standalone raw-feature model (default: Random Forest) for Kalshi BTC 15M."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline

DATA_PATH = "kalshi_btc15m_dataset_30k.csv"
MODEL_OUTPUT = "raw_feature_model.pkl"
TARGET = "final_outcome_yes"
TEST_SIZE = 0.20

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
    "momentum_acceleration",
    "flip_count_5m",
    "return_1m_x_inv_time",
    "return_3m_x_inv_time",
    "volatility_5m_x_inv_time",
    "bid_ask_spread",
    "volume_acceleration",
    "trade_intensity",
    "rsi_14",
    "session",
    "distance_from_strike",
    "outcome_rate_bucket",
    "return_5m_ratio",
]


def confirm_overwrite(output_path: Path) -> bool:
    """Ask user for overwrite confirmation if output file already exists."""
    if not output_path.exists():
        return True

    print(f"Warning: '{output_path}' already exists.")
    answer = input("Overwrite existing model file? (y/n): ").strip().lower()
    return answer == "y"


def load_and_prepare_data(csv_path: Path) -> pd.DataFrame:
    """Load dataset and apply required preprocessing for temporal training."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at path: {csv_path}")

    df = pd.read_csv(csv_path, dtype={"series_ticker": str})

    required_columns = set(RAW_FEATURES + [TARGET, "price_now", "close_ts", "market_ticker", "entry_bucket"])
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["avg_trade_price_1m"] = df["avg_trade_price_1m"].fillna(df["price_now"])
    df["avg_trade_price_3m"] = df["avg_trade_price_3m"].fillna(df["price_now"])
    df = df.sort_values(by=["close_ts", "market_ticker", "entry_bucket"]).reset_index(drop=True)

    return df


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data by market_ticker to prevent data leakage.

    Markets are sorted by close_ts and split 80/20 so no market appears
    in both sets. A 15-minute embargo gap is enforced between the last
    training market and the first test market.
    """
    if not 0 < TEST_SIZE < 1:
        raise ValueError("TEST_SIZE must be between 0 and 1.")

    if "market_ticker" not in df.columns:
        # Fallback to row-level split if no market_ticker column
        split_idx = int(len(df) * (1 - TEST_SIZE))
        return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()

    # Get unique markets sorted by their earliest close_ts
    market_order = (
        df.groupby("market_ticker")["close_ts"]
        .min()
        .sort_values()
        .index
        .tolist()
    )
    n_test_markets = max(1, int(len(market_order) * TEST_SIZE))
    n_train_markets = len(market_order) - n_test_markets
    if n_train_markets <= 0:
        raise ValueError("Not enough markets for train/test split.")

    train_tickers = set(market_order[:n_train_markets])
    test_tickers = set(market_order[n_train_markets:])

    train_df = df[df["market_ticker"].isin(train_tickers)].copy()
    test_df = df[df["market_ticker"].isin(test_tickers)].copy()

    # Enforce 15-minute embargo between last train and first test
    max_train_ts = train_df["close_ts"].max()
    min_test_ts = test_df["close_ts"].min()
    if min_test_ts - max_train_ts < 900:
        # Drop test markets that start within the embargo window
        embargo_cutoff = max_train_ts + 900
        test_df = test_df[test_df["close_ts"] >= embargo_cutoff].copy()

    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError("Train/test split resulted in an empty partition.")

    return train_df, test_df


def make_xgboost() -> Pipeline:
    """Median-impute + XGBoost pipeline (for experiments / comparison)."""
    from xgboost import XGBClassifier

    estimator = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", estimator),
        ]
    )


def build_pipeline() -> Pipeline:
    """Construct median-impute + Random Forest classification pipeline."""
    return Pipeline(
        steps=[
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


def compute_metrics(y_true: pd.Series, y_pred: pd.Series, y_proba: pd.Series) -> dict[str, float]:
    """Compute standard binary classification metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "brier_score": float(brier_score_loss(y_true, y_proba)),
        "log_loss": float(log_loss(y_true, y_proba)),
    }


def print_class_distribution(name: str, y: pd.Series) -> None:
    """Print class distribution for a target vector."""
    counts = y.value_counts(dropna=False).sort_index()
    pct = y.value_counts(normalize=True, dropna=False).sort_index()
    print(f"\n{name} class distribution:")
    for cls in counts.index:
        print(f"  class={cls}: count={int(counts[cls])}, pct={pct[cls]:.4f}")


def print_saved_metadata(metadata: dict) -> None:
    """Print summary table of saved model metadata."""
    print("\nSaved metadata summary:")
    print("-" * 72)
    rows = [
        ("model_type", metadata.get("model_type", "—")),
        ("trained_at", metadata["trained_at"]),
        ("feature_count", len(metadata["features"])),
        ("n_train", metadata["n_train"]),
        ("n_test", metadata["n_test"]),
        ("accuracy", f'{metadata["test_metrics"]["accuracy"]:.6f}'),
        ("roc_auc", f'{metadata["test_metrics"]["roc_auc"]:.6f}'),
        ("brier_score", f'{metadata["test_metrics"]["brier_score"]:.6f}'),
        ("log_loss", f'{metadata["test_metrics"]["log_loss"]:.6f}'),
    ]
    for key, value in rows:
        print(f"{key:<20} | {value}")
    print("-" * 72)


def train_model(data_path: str, model_output: str) -> dict:
    """Train, evaluate, and save model artifact. Returns metadata bundle."""
    print(f"Training environment: Python {sys.version}")
    print(f"sklearn version: {sklearn.__version__}")
    print(f"numpy version: {np.__version__}")

    data_path_obj = Path(data_path)
    model_output_obj = Path(model_output)

    if model_output_obj.exists():
        print(f"Warning: '{model_output_obj}' already exists.")
        answer = input("Overwrite existing model file? (y/n): ").strip().lower()
        if answer != "y":
            print("Training aborted. Existing model was not overwritten.")
            return {}

    df = load_and_prepare_data(data_path_obj)
    train_df, test_df = temporal_split(df)

    x_train = train_df[RAW_FEATURES]
    y_train = train_df[TARGET]
    x_test = test_df[RAW_FEATURES]
    y_test = test_df[TARGET]

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)
    y_proba = pipeline.predict_proba(x_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_proba)

    print("\nTest metrics:")
    print(f"  Accuracy   : {metrics['accuracy']:.6f}")
    print(f"  ROC-AUC    : {metrics['roc_auc']:.6f}")
    print(f"  Brier Score: {metrics['brier_score']:.6f}")
    print(f"  Log Loss   : {metrics['log_loss']:.6f}")

    print_class_distribution("Train", y_train)
    print_class_distribution("Test", y_test)
    print(f"\nFeature count confirmation: {len(RAW_FEATURES)}")

    artifact = {
        "model": pipeline,
        "features": RAW_FEATURES,
        "model_type": "RandomForest",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "test_metrics": metrics,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "sklearn_version": sklearn.__version__,
        "python_version": sys.version,
    }

    joblib.dump(artifact, model_output_obj)
    print(f"\nModel artifact saved to: {model_output_obj.resolve()}")
    print_saved_metadata(artifact)
    return artifact


if __name__ == "__main__":
    train_model(DATA_PATH, MODEL_OUTPUT)
