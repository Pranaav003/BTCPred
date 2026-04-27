"""Thread-safe model loading and inference helpers."""

from __future__ import annotations

import threading
from pathlib import Path
import logging

import joblib
import pandas as pd
import sklearn

_MODEL_BUNDLE: dict | None = None
_MODEL_LOCK = threading.Lock()
logger = logging.getLogger(__name__)


def load_model(model_path: str = "raw_feature_model.pkl") -> dict:
    """Load model bundle from disk and validate required keys."""
    path = Path(model_path)
    if not path.exists():
        raise RuntimeError(
            f"Model file not found at '{path}'. Run 'python train_raw_model.py' first."
        )

    bundle = joblib.load(path)
    saved_ver = bundle.get("sklearn_version", "unknown") if isinstance(bundle, dict) else "unknown"
    current_ver = sklearn.__version__
    if saved_ver != current_ver:
        logger.warning(
            "sklearn version mismatch: model saved with %s, running %s. Predictions may be unreliable. Retrain recommended.",
            saved_ver,
            current_ver,
        )
    required = {"model", "features", "trained_at", "test_metrics"}
    missing = required - set(bundle.keys())
    if missing:
        raise RuntimeError(f"Model bundle missing required keys: {sorted(missing)}")
    if isinstance(bundle, dict) and "model_type" not in bundle:
        bundle = {**bundle, "model_type": "Unknown"}
    return bundle


def get_model() -> dict:
    """Return cached model bundle, loading once in a thread-safe way."""
    global _MODEL_BUNDLE

    if _MODEL_BUNDLE is not None:
        return _MODEL_BUNDLE

    with _MODEL_LOCK:
        if _MODEL_BUNDLE is None:
            _MODEL_BUNDLE = load_model()
    return _MODEL_BUNDLE


def predict_proba_raw(feature_dict: dict) -> float:
    """Predict class-1 probability using provided raw feature dictionary."""
    bundle = get_model()
    model = bundle["model"]
    features = bundle["features"]

    ordered_values = [float(feature_dict.get(feature, 0.0) or 0.0) for feature in features]
    frame = pd.DataFrame([ordered_values], columns=features)
    proba_yes = model.predict_proba(frame)[:, 1][0]
    return float(proba_yes)
