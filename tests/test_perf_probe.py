# tests/test_perf_probe.py
import importlib.util
import os
import pathlib

import pytest

# The perf probe runs the REAL hot path (predict_proba_raw), which needs the
# model artifact. That .pkl is gitignored, so it is absent in a fresh CI
# checkout — skip there, matching the MODEL_PRESENT convention in
# tests/test_e2e_pipeline.py.
MODEL_PRESENT = os.path.exists("raw_feature_model.pkl")

_spec = importlib.util.spec_from_file_location(
    "perf_probe", pathlib.Path("scripts/perf_probe.py"))
pp = importlib.util.module_from_spec(_spec)


def _load():
    _spec.loader.exec_module(pp)
    return pp


@pytest.mark.skipif(not MODEL_PRESENT, reason="raw_feature_model.pkl artifact required")
def test_measure_returns_all_hotpath_keys():
    mod = _load()
    m = mod.measure(n=3)  # tiny n: this runs the REAL probe once, keep it fast
    for key in ("predict_proba_raw_ms", "api_control_state_ms",
                "api_settings_ms", "api_health_ms"):
        assert key in m
        assert isinstance(m[key], float) and m[key] >= 0.0
