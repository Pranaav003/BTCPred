"""Thread-safe model loading and inference helpers."""

from __future__ import annotations

import gzip
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
    """Try to load model from disk. Returns None if file missing or unreadable.

    A corrupt/unreadable .pkl must NOT crash load_model — returning None lets the
    DB fallback (_load_from_db) run, which is a legitimate second source.
    """
    path = Path(model_path)
    if not path.exists():
        return None
    try:
        bundle = joblib.load(path)
    except Exception:
        logger.warning("Failed to load model from disk %s; falling back to DB", path, exc_info=True)
        return None
    if not isinstance(bundle, dict):
        return None
    return bundle


def _decompress(data: bytes) -> bytes:
    """Decompress gzip data, or return raw bytes if not gzipped."""
    try:
        return gzip.decompress(data)
    except (gzip.BadGzipFile, OSError):
        # Not gzipped — legacy uncompressed storage
        return data


def _load_from_db() -> dict | None:
    """Try to load model from PostgreSQL ModelArtifact table. Returns None if not stored."""
    try:
        from app.models import ModelArtifact
        from app.extensions import db

        artifact = ModelArtifact.query.filter_by(name="default").first()
        if artifact is None or artifact.data is None:
            return None
        raw = _decompress(artifact.data)
        bundle = joblib.load(io.BytesIO(raw))
        if not isinstance(bundle, dict):
            return None
        logger.info(
            "Loaded model from DB (uploaded %s, %d bytes stored, %d bytes decompressed)",
            artifact.uploaded_at,
            len(artifact.data),
            len(raw),
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
    """Read a .pkl from disk, gzip-compress, and persist to the ModelArtifact table.

    Compression shrinks the 15MB pkl to ~3-5MB, avoiding SSL connection
    timeouts on Render's starter PostgreSQL.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    from app.models import ModelArtifact
    from app.extensions import db

    raw = path.read_bytes()
    bundle = joblib.load(io.BytesIO(raw))
    if not isinstance(bundle, dict):
        raise ValueError("Model file does not contain a valid bundle dict")

    compressed = gzip.compress(raw, compresslevel=6)

    metrics = bundle.get("test_metrics", {})
    artifact = ModelArtifact.query.filter_by(name="default").first()
    if artifact is None:
        artifact = ModelArtifact(name="default", data=compressed)
        db.session.add(artifact)
    else:
        artifact.data = compressed
        artifact.uploaded_at = None  # let default kick in

    artifact.size_bytes = len(compressed)
    artifact.model_type = bundle.get("model_type")
    artifact.accuracy = metrics.get("accuracy")
    db.session.commit()

    logger.info(
        "Model saved to DB: %d bytes raw → %d bytes compressed (%.0f%% reduction), accuracy=%.4f",
        len(raw), len(compressed), 100 * (1 - len(compressed) / len(raw)),
        metrics.get("accuracy", 0),
    )
    return {
        "size_bytes": len(raw),
        "size_bytes_stored": len(compressed),
        "model_type": bundle.get("model_type"),
        "accuracy": metrics.get("accuracy"),
        "trained_at": bundle.get("trained_at"),
    }


def load_model_to_disk_from_db(model_path: str = "raw_feature_model.pkl") -> bool:
    """If model is in DB but not on disk, decompress and write it to disk (for train_raw_model.py compat)."""
    path = Path(model_path)
    if path.exists():
        return False

    from app.models import ModelArtifact

    artifact = ModelArtifact.query.filter_by(name="default").first()
    if artifact is None or artifact.data is None:
        return False

    raw = _decompress(artifact.data)
    path.write_bytes(raw)
    logger.info("Wrote model from DB to disk: %s (%d bytes)", path, len(raw))
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
    try:
        proba_yes = model.predict_proba(frame)[:, 1][0]
    except Exception as exc:
        # FAIL LOUD: a broken model must never yield a fabricated probability —
        # trading on a made-up number is worse than skipping the cycle. Re-raise
        # as RuntimeError (the module's model-unusable convention) so the sole
        # caller's `except RuntimeError` in signal_engine skips the cycle safely,
        # preserving the original cause for diagnosis.
        logger.exception(
            "Model inference failed (%d features); refusing to fabricate a probability",
            len(features),
        )
        raise RuntimeError("Model inference failed in predict_proba_raw") from exc
    return float(proba_yes)
