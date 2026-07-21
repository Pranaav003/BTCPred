# scripts/perf_probe.py
"""Measure median latency (ms) of the app's hot paths. Warmup + median of N runs."""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

# Ensure repo root is importable regardless of how this is invoked (as a script,
# scripts/ is sys.path[0]; `import app` needs the repo root).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _bench(fn, n: int, warmup: int = 5) -> float:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t) * 1000.0)
    return float(statistics.median(samples))


def measure(n: int = 100) -> dict:
    from app import create_app
    from app.model_loader import predict_proba_raw

    feats = {"seconds_to_close": 100, "return_1m": 5.0, "return_3m": 10.0,
             "return_5m": 8.0, "volatility_5m": 30.0, "rsi_14": 55.0, "price_now": 0.5}
    predict_proba_raw(feats)  # warm the model load out of the timing

    app = create_app("testing")
    client = app.test_client()

    return {
        "predict_proba_raw_ms": _bench(lambda: predict_proba_raw(feats), n),
        "api_control_state_ms": _bench(lambda: client.get("/api/control/state"), n),
        "api_settings_ms": _bench(lambda: client.get("/api/settings"), n),
        "api_health_ms": _bench(lambda: client.get("/api/health"), n),
    }
