# tests/test_perf_probe.py
import importlib.util
import pathlib

_spec = importlib.util.spec_from_file_location(
    "perf_probe", pathlib.Path("scripts/perf_probe.py"))
pp = importlib.util.module_from_spec(_spec)


def _load():
    _spec.loader.exec_module(pp)
    return pp


def test_measure_returns_all_hotpath_keys():
    mod = _load()
    m = mod.measure(n=3)  # tiny n: this runs the REAL probe once, keep it fast
    for key in ("predict_proba_raw_ms", "api_control_state_ms",
                "api_settings_ms", "api_health_ms"):
        assert key in m
        assert isinstance(m[key], float) and m[key] >= 0.0
