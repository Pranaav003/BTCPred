"""Train and compare multiple model types on the same dataset."""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = "kalshi_btc15m_dataset_30k.csv"
TARGET = "final_outcome_yes"
TEST_SIZE = 0.20
MODEL_OUTPUT = "raw_feature_model.pkl"
RESULTS_OUTPUT = "model_comparison_results.csv"

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


def load_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
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
    if "market_ticker" not in df.columns:
        split_idx = int(len(df) * (1 - TEST_SIZE))
        if split_idx <= 0 or split_idx >= len(df):
            raise ValueError("Dataset too small for configured TEST_SIZE.")
        return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()

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

    max_train_ts = train_df["close_ts"].max()
    min_test_ts = test_df["close_ts"].min()
    if min_test_ts - max_train_ts < 900:
        embargo_cutoff = max_train_ts + 900
        test_df = test_df[test_df["close_ts"] >= embargo_cutoff].copy()

    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError("Train/test split resulted in an empty partition.")

    return train_df, test_df


def calibration_error(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bucket_ids = np.digitize(y_proba, bins, right=True)
    diffs: list[float] = []
    for b in range(1, n_bins + 1):
        mask = bucket_ids == b
        if np.sum(mask) == 0:
            continue
        pred_mean = float(np.mean(y_proba[mask]))
        actual_rate = float(np.mean(y_true[mask]))
        diffs.append(abs(pred_mean - actual_rate))
    return float(np.mean(diffs)) if diffs else 0.0


def build_model_specs(scale_pos_weight: float = 1.0) -> list[tuple[str, Pipeline]]:
    specs: list[tuple[str, Pipeline]] = []
    from xgboost import XGBClassifier

    specs.append(
        (
            "XGBoost",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        XGBClassifier(
                            n_estimators=300,
                            max_depth=4,
                            learning_rate=0.05,
                            subsample=0.9,
                            colsample_bytree=0.9,
                            random_state=42,
                            objective="binary:logistic",
                            eval_metric="logloss",
                            scale_pos_weight=scale_pos_weight,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
        )
    )

    try:
        from lightgbm import LGBMClassifier

        specs.append(
            (
                "LightGBM",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        (
                            "model",
                            LGBMClassifier(
                                n_estimators=300,
                                max_depth=4,
                                learning_rate=0.05,
                                subsample=0.9,
                                colsample_bytree=0.9,
                                random_state=42,
                                verbose=-1,
                            ),
                        ),
                    ]
                ),
            )
        )
    except ImportError:
        print("LightGBM not installed, skipping. Run: pip install lightgbm")

    specs.append(
        (
            "RandomForest",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=300,
                            max_depth=8,
                            min_samples_leaf=10,
                            class_weight="balanced",
                            random_state=42,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
        )
    )

    specs.append(
        (
            "LogisticRegression",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")),
                ]
            ),
        )
    )

    specs.append(
        (
            "GradientBoosting",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        GradientBoostingClassifier(
                            n_estimators=200,
                            max_depth=4,
                            learning_rate=0.05,
                            random_state=42,
                        ),
                    ),
                ]
            ),
        )
    )

    # Stacking ensemble: RF, XGBoost, LR as base estimators with LR meta-learner
    stacking = StackingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=10,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )),
            ("xgb", XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                objective="binary:logistic",
                eval_metric="logloss",
                scale_pos_weight=scale_pos_weight,
                n_jobs=-1,
            )),
            ("lr", LogisticRegression(
                max_iter=2000,
                C=1.0,
                class_weight="balanced",
            )),
        ],
        final_estimator=LogisticRegression(max_iter=2000, class_weight="balanced"),
        cv=3,
        n_jobs=-1,
    )
    specs.append(
        (
            "StackingEnsemble",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", stacking),
                ]
            ),
        )
    )
    return specs


def train_and_evaluate(
    name: str, pipeline: Pipeline, x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, y_test: pd.Series
) -> dict:
    started = time.perf_counter()
    pipeline.fit(x_train, y_train)
    train_seconds = time.perf_counter() - started

    y_proba = pipeline.predict_proba(x_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    result = {
        "model": name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "brier_score": float(brier_score_loss(y_test, y_proba)),
        "log_loss": float(log_loss(y_test, y_proba)),
        "calibration_error": float(calibration_error(y_test.to_numpy(), y_proba, n_bins=10)),
        "train_seconds": float(train_seconds),
        "pipeline": pipeline,
    }
    return result


def mark_best(df: pd.DataFrame, column: str, higher_is_better: bool) -> set[str]:
    if higher_is_better:
        best_value = df[column].max()
    else:
        best_value = df[column].min()
    return set(df.loc[df[column] == best_value, "model"].tolist())


def print_comparison_table(results_df: pd.DataFrame) -> None:
    col_best = {
        "accuracy": mark_best(results_df, "accuracy", True),
        "roc_auc": mark_best(results_df, "roc_auc", True),
        "brier_score": mark_best(results_df, "brier_score", False),
        "log_loss": mark_best(results_df, "log_loss", False),
        "calibration_error": mark_best(results_df, "calibration_error", False),
        "train_seconds": mark_best(results_df, "train_seconds", False),
    }

    print("\nModel comparison (sorted by ROC-AUC desc):")
    print("-" * 108)
    print(
        f"{'Model':<20} | {'Accuracy':>9} | {'ROC-AUC':>8} | {'Brier':>8} | "
        f"{'LogLoss':>8} | {'Calibration':>11} | {'Train Time':>10}"
    )
    print("-" * 108)
    for _, row in results_df.iterrows():
        model = row["model"]
        acc = f"{row['accuracy']:.4f}{'*' if model in col_best['accuracy'] else ''}"
        auc = f"{row['roc_auc']:.4f}{'*' if model in col_best['roc_auc'] else ''}"
        brier = f"{row['brier_score']:.4f}{'*' if model in col_best['brier_score'] else ''}"
        ll = f"{row['log_loss']:.4f}{'*' if model in col_best['log_loss'] else ''}"
        cal = f"{row['calibration_error']:.4f}{'*' if model in col_best['calibration_error'] else ''}"
        secs = f"{row['train_seconds']:.2f}{'*' if model in col_best['train_seconds'] else ''}s"
        print(f"{model:<20} | {acc:>9} | {auc:>8} | {brier:>8} | {ll:>8} | {cal:>11} | {secs:>10}")
    print("-" * 108)
    print("* best value in column")


def top_importances(result: dict, features: list[str], top_n: int = 10) -> list[tuple[str, float]]:
    pipeline: Pipeline = result["pipeline"]
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return []
    importances = model.feature_importances_
    pairs = list(zip(features, importances))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:top_n]


def print_feature_importance_sections(results: list[dict], features: list[str]) -> None:
    wanted = {"XGBoost", "RandomForest"}
    top_maps: dict[str, list[tuple[str, float]]] = {}
    for result in results:
        if result["model"] in wanted:
            tops = top_importances(result, features, top_n=10)
            if tops:
                top_maps[result["model"]] = tops

    for model_name in ("XGBoost", "RandomForest"):
        if model_name not in top_maps:
            continue
        print(f"\nTop 10 feature importances — {model_name}:")
        for idx, (feat, score) in enumerate(top_maps[model_name], start=1):
            print(f"  {idx:>2}. {feat:<28} {score:.6f}")

    if "XGBoost" in top_maps and "RandomForest" in top_maps:
        xgb_feats = {f for f, _ in top_maps["XGBoost"]}
        rf_feats = {f for f, _ in top_maps["RandomForest"]}
        overlap = sorted(xgb_feats & rf_feats)
        print("\nImportance agreement (XGBoost vs RandomForest):")
        if overlap:
            print(f"  Shared top features ({len(overlap)}): {', '.join(overlap)}")
        else:
            print("  No overlap in top-10 importance lists.")


def choose_recommended_model(results_df: pd.DataFrame) -> tuple[str, str]:
    best_row = results_df.iloc[0]
    name = str(best_row["model"])
    reason = (
        f"highest ROC-AUC ({best_row['roc_auc']:.4f}) with "
        f"Brier={best_row['brier_score']:.4f}, calibration error={best_row['calibration_error']:.4f}."
    )
    return name, reason


def maybe_retrain_best(best_row: pd.Series, n_train: int, n_test: int) -> None:
    artifact = {
        "model": best_row["pipeline"],
        "features": RAW_FEATURES,
        "trained_at": datetime.utcnow().isoformat(),
        "test_metrics": {
            "accuracy": float(best_row["accuracy"]),
            "roc_auc": float(best_row["roc_auc"]),
            "brier_score": float(best_row["brier_score"]),
            "log_loss": float(best_row["log_loss"]),
            "calibration_error": float(best_row["calibration_error"]),
            "model_name": str(best_row["model"]),
        },
        "n_train": int(n_train),
        "n_test": int(n_test),
    }
    joblib.dump(artifact, MODEL_OUTPUT)
    print(f"\nSaved best model artifact to {Path(MODEL_OUTPUT).resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare model families on Kalshi BTC dataset.")
    parser.add_argument("--retrain", action="store_true", help="Retrain raw_feature_model.pkl with best model.")
    args = parser.parse_args()

    df = load_data(Path(DATA_PATH))
    train_df, test_df = temporal_split(df)
    x_train = train_df[RAW_FEATURES]
    y_train = train_df[TARGET]
    x_test = test_df[RAW_FEATURES]
    y_test = test_df[TARGET]

    # Compute class imbalance ratio for scale_pos_weight (XGBoost)
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    specs = build_model_specs(scale_pos_weight=scale_pos_weight)
    if not specs:
        raise RuntimeError("No models available to compare.")

    results: list[dict] = []
    for name, pipeline in specs:
        print(f"Training {name}...")
        result = train_and_evaluate(name, pipeline, x_train, y_train, x_test, y_test)
        results.append(result)

    results_df = pd.DataFrame(results).sort_values(by="roc_auc", ascending=False).reset_index(drop=True)
    print_comparison_table(results_df)
    print_feature_importance_sections(results, RAW_FEATURES)

    csv_df = results_df.drop(columns=["pipeline"]).copy()
    csv_df.to_csv(RESULTS_OUTPUT, index=False)
    print(f"\nSaved comparison CSV: {Path(RESULTS_OUTPUT).resolve()}")

    recommended_name, reason = choose_recommended_model(results_df)
    print(f"\nRecommended model: {recommended_name}")
    print(f"Reason: {reason}")

    if args.retrain:
        maybe_retrain_best(results_df.iloc[0], len(train_df), len(test_df))


if __name__ == "__main__":
    main()

