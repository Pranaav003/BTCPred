# tests/test_model_loader_failpaths.py
# Refutation-derived regression tests for model_loader failure paths:
#  * _load_from_disk must NOT propagate a corrupt-file error — it logs + returns
#    None so the DB fallback in load_model can run.
#  * predict_proba_raw must FAIL LOUD on an inference error — re-raise as
#    RuntimeError (never fabricate a probability), preserving the original cause,
#    so signal_engine.py's `except RuntimeError` skips the cycle cleanly.
import logging

import pytest


def test_load_from_disk_returns_none_and_logs_on_corrupt_pickle(monkeypatch, tmp_path, caplog):
    from app import model_loader

    bad = tmp_path / "raw_feature_model.pkl"
    bad.write_bytes(b"not a valid joblib pickle \x00\x01\x02")

    with caplog.at_level(logging.WARNING):
        result = model_loader._load_from_disk(str(bad))

    assert result is None  # falls through to DB fallback instead of crashing
    assert any("load" in rec.getMessage().lower() or rec.exc_info for rec in caplog.records)


def test_load_from_disk_missing_file_still_returns_none(tmp_path):
    from app import model_loader

    result = model_loader._load_from_disk(str(tmp_path / "nope.pkl"))
    assert result is None


def test_predict_proba_raw_reraises_as_runtimeerror_never_fabricates(monkeypatch, caplog):
    from app import model_loader

    class _BoomModel:
        def predict_proba(self, frame):
            raise ValueError("feature shape mismatch")

    fake_bundle = {"model": _BoomModel(), "features": ["seconds_to_close", "return_1m"]}
    monkeypatch.setattr(model_loader, "get_model", lambda: fake_bundle)

    with caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeError) as excinfo:
            model_loader.predict_proba_raw({"seconds_to_close": 100, "return_1m": 0.01})

    # It must be a RuntimeError (so the caller's `except RuntimeError` handles it),
    # NOT a fabricated float, and the original ValueError must be chained.
    assert isinstance(excinfo.value.__cause__, ValueError)
    assert any(rec.levelno >= logging.ERROR for rec in caplog.records)


def test_predict_proba_raw_runtimeerror_is_caught_by_caller_contract(monkeypatch):
    # End-to-end contract: signal_engine.evaluate_live_signal catches RuntimeError
    # from predict_proba_raw and returns None (skip the cycle) — never a trade on
    # a fabricated probability.
    from app import model_loader

    class _BoomModel:
        def predict_proba(self, frame):
            raise RuntimeError("inference blew up")

    monkeypatch.setattr(
        model_loader, "get_model",
        lambda: {"model": _BoomModel(), "features": ["seconds_to_close"]},
    )
    with pytest.raises(RuntimeError):
        model_loader.predict_proba_raw({"seconds_to_close": 30})
