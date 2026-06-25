"""Thread-safe model loading and inference helpers."""

from __future__ import annotations

import io
import threading
from pathlib import Path
import logging

import joblib
import pandas as pd
import sklearn

_MODEL_BUNDLE: dict | None = None
_MODEL_LOCK = threading.Lock()
logger = logging.getLogger(__name__)


def _load_from_disk(model_path: str = "raw_feature_model.pkl") -> dict | None:
    """Try to load model from disk. Returns None if file missing."""
    path = Path(model_path)
    if not path.exists():
        return None
    bundle = joblib.load(path)
    if not isinstance(bundle, dict):
        return None
    return bundle


def _load_from_db() -> dict | None:
    """Try to load model from PostgreSQL ModelArtifact table. Returns None if not stored."""
    try:
        from app.models import ModelArtifact
        from app.extensions import db

        artifact = ModelArtifact.query.filter_by(name="default").first()
        if artifact is None or artifact.data is None:
            return None
        bundle = joblib.load(io.BytesIO(artifact.data))
        if not isinstance(bundle, dict):
            return None
        logger.info(
            "Loaded model from DB (uploaded %s, %d bytes)",
            artifact.uploaded_at,
            len(artifact.data),
        )
        return bundle
    except Exception as exc:
        logger.warning("Could not load model from DB: %s", exc)
        return None


def _validate_bundle(bundle: dict) -> dict:
    """Validate required keys and warn about sklearn version mismatch."""
    saved_ver = bundle.get("sklearn_version", "unknown")
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
    if "model_type" not in bundle:
        bundle = {**bundle, "model_type": "Unknown"}
    return bundle


def load_model(model_path: str = "raw_feature_model.pkl") -> dict:
    """Load model bundle from disk, falling back to DB storage."""
    # Try disk first (faster, no DB query)
    bundle = _load_from_disk(model_path)
    source = "disk"

    # Fall back to DB (survives Render deploys)
    if bundle is None:
        bundle = _load_from_db()
        source = "database"

    if bundle is None:
        raise RuntimeError(
            "No model found on disk or in database. "
            "Upload one via /api/model/upload or run train_raw_model.py"
        )

    logger.info("Model loaded from %s", source)
    return _validate_bundle(bundle)


def get_model() -> dict:
    """Return cached model bundle, loading once in a thread-safe way."""
    global _MODEL_BUNDLE

    if _MODEL_BUNDLE is not None:
        return _MODEL_BUNDLE

    with _MODEL_LOCK:
        if _MODEL_BUNDLE is None:
            _MODEL_BUNDLE = load_model()
    return _MODEL_BUNDLE


def clear_model_cache() -> None:
    """Clear the cached model bundle so the next inference reloads from disk."""
    global _MODEL_BUNDLE
    with _MODEL_LOCK:
        _MODEL_BUNDLE = None
    logger.info("Model cache cleared — next prediction will reload from disk/DB")


def save_model_to_db(model_path: str = "raw_feature_model.pkl") -> dict:
    """Read a .pkl from disk and persist it to the ModelArtifact table."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    from app.models import ModelArtifact
    from app.extensions import db

    data = path.read_bytes()
    bundle = joblib.load(io.BytesIO(data))
    if not isinstance(bundle, dict):
        raise ValueError("Model file does not contain a valid bundle dict")

    metrics = bundle.get("test_metrics", {})
    artifact = ModelArtifact.query.filter_by(name="default").first()
    if artifact is None:
        artifact = ModelArtifact(name="default", data=data)
        db.session.add(artifact)
    else:
        artifact.data = data
        artifact.uploaded_at = None  # let default kick in

    artifact.size_bytes = len(data)
    artifact.model_type = bundle.get("model_type")
    artifact.accuracy = metrics.get("accuracy")
    db.session.commit()

    logger.info("Model saved to DB: %d bytes, accuracy=%.4f", len(data), metrics.get("accuracy", 0))
    return {
        "size_bytes": len(data),
        "model_type": bundle.get("model_type"),
        "accuracy": metrics.get("accuracy"),
        "trained_at": bundle.get("trained_at"),
    }


def load_model_to_disk_from_db(model_path: str = "raw_feature_model.pkl") -> bool:
    """If model is in DB but not on disk, write it to disk (for train_raw_model.py compat)."""
    path = Path(model_path)
    if path.exists():
        return False

    from app.models import ModelArtifact

    artifact = ModelArtifact.query.filter_by(name="default").first()
    if artifact is None or artifact.data is None:
        return False

    path.write_bytes(artifact.data)
    logger.info("Wrote model from DB to disk: %s (%d bytes)", path, len(artifact.data))
    return True


def predict_proba_raw(feature_dict: dict) -> float:
    """Predict class-1 probability using provided raw feature dictionary."""
    bundle = get_model()
    model = bundle["model"]
    features = bundle["features"]

    ordered_values = []
    for feature in features:
        val = feature_dict.get(feature, 0.0)
        if val is None or val == "":
            logger.warning("Missing or empty feature '%s' — defaulting to 0.0", feature)
            val = 0.0
        ordered_values.append(float(val or 0.0))
    frame = pd.DataFrame([ordered_values], columns=features)
    proba_yes = model.predict_proba(frame)[:, 1][0]
    return float(proba_yes)
