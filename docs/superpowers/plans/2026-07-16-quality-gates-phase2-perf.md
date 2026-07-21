# Quality Gates Phase 2 (Performance Baselines) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add performance baselines for the app's hot paths, gated on-demand + in CI at +10% (ratchet-down), stored in `test-baseline.json` under `perf` — separate from the deterministic every-turn gate.

**Architecture:** A lightweight `scripts/perf_probe.py` (`measure()` → median ms per hot path); a pure `perf_ratchet` + a `--perf` mode in `scripts/check_quality.py`; CI runs perf non-blocking; pre-commit exposes perf as a manual hook.

**Tech Stack:** Python 3.13 venv, pytest, stdlib `time`/`statistics`.

## Global Constraints

- Use the venv for everything: `.venv/bin/python -m pytest`, `.venv/bin/python scripts/check_quality.py`. Never bare tools.
- Perf tolerance is **+10%**: a metric is OK iff `current <= baseline * 1.10`. Perf ratchets **down** (baseline becomes `min`).
- `--perf` is **separate** from `--check-only` — the deterministic Stop-hook gate must remain test/coverage/ruff/mypy only (do not add perf to it).
- **Never run the real perf probe inside a unit test** — monkeypatch `_run_perf`/`measure` (slow + noisy). One `@pytest.mark`-free shape smoke test that runs `measure(n=small)` once is allowed (Task 1).
- No module-level `from app import ...` in test files.
- Every task ends green (full suite) and commits. Branch: `feature/quality-gates-phase2-perf`.

---

### Task 1: Perf probe module

**Files:**
- Create: `scripts/perf_probe.py`
- Test: `tests/test_perf_probe.py`

**Interfaces:**
- Produces: `measure(n: int = 100) -> dict` with float-ms values under keys `predict_proba_raw_ms`, `api_control_state_ms`, `api_settings_ms`, `api_health_ms`.

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_perf_probe.py -v`
Expected: FAIL — `scripts/perf_probe.py` does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_perf_probe.py -v`
Expected: PASS (1). Then full suite `.venv/bin/python -m pytest -q` → green (prior + 1).

- [ ] **Step 5: Commit**

```bash
git add scripts/perf_probe.py tests/test_perf_probe.py
git commit -m "feat(perf): hot-path latency probe (measure median ms)"
```

---

### Task 2: perf ratchet + `--perf` mode + seed baseline

**Files:**
- Modify: `scripts/check_quality.py` (add `PERF_TOLERANCE`, `perf_ok`, `perf_ratchet`, `_run_perf`, `--perf` in `main`)
- Modify: `test-baseline.json` (gains a `perf` block via seeding)
- Test: `tests/test_check_quality.py` (pure ratchet tests + monkeypatched `--perf` CLI)

**Interfaces:**
- Consumes: `measure()` from `scripts/perf_probe.py`.
- Produces: `PERF_TOLERANCE = 1.10`; `perf_ok(current, baseline, tol=PERF_TOLERANCE) -> bool` (= `current <= baseline*tol`); `perf_ratchet(current: dict, baseline: dict, tol=PERF_TOLERANCE) -> tuple[bool, dict]` (ok=False if ANY metric exceeds band; on ok, each key → `min(current, baseline)`; unknown keys seed); `_run_perf() -> dict`; `main(["--perf"])`.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_check_quality.py
def test_perf_ok_band():
    assert cq.perf_ok(44.0, 43.0) is True     # within +10%
    assert cq.perf_ok(47.3, 43.0) is True      # ~ exactly 43*1.1
    assert cq.perf_ok(48.0, 43.0) is False     # over band


def test_perf_ratchet_within_band_ratchets_down():
    base = {"predict_proba_raw_ms": 43.0, "api_health_ms": 0.2}
    ok, new = cq.perf_ratchet({"predict_proba_raw_ms": 40.0, "api_health_ms": 0.2}, base)
    assert ok is True
    assert new["predict_proba_raw_ms"] == 40.0   # improved -> min
    assert new["api_health_ms"] == 0.2


def test_perf_ratchet_over_band_fails_and_holds():
    base = {"predict_proba_raw_ms": 43.0}
    ok, new = cq.perf_ratchet({"predict_proba_raw_ms": 48.0}, base)  # >43*1.1
    assert ok is False and new == base


def test_perf_ratchet_seeds_unknown_metric():
    ok, new = cq.perf_ratchet({"new_ms": 5.0}, {})
    assert ok is True and new["new_ms"] == 5.0


def test_perf_cli_within_band_returns_zero(monkeypatch, tmp_path):
    import json
    bl = tmp_path / "tb.json"
    bl.write_text(json.dumps({"tests_passed": 1, "coverage_pct": 0.0,
                              "ruff_violations": 0, "mypy_errors": 0,
                              "perf": {"api_health_ms": 0.2}}))
    monkeypatch.setattr(cq, "BASELINE_PATH", bl)
    monkeypatch.setattr(cq, "_run_perf", lambda: {"api_health_ms": 0.19})
    assert cq.main(["--perf"]) == 0
    assert json.loads(bl.read_text())["perf"]["api_health_ms"] == 0.19  # ratcheted down


def test_perf_cli_over_band_returns_one(monkeypatch, tmp_path):
    import json
    bl = tmp_path / "tb.json"
    bl.write_text(json.dumps({"tests_passed": 1, "coverage_pct": 0.0,
                              "ruff_violations": 0, "mypy_errors": 0,
                              "perf": {"api_health_ms": 0.2}}))
    monkeypatch.setattr(cq, "BASELINE_PATH", bl)
    monkeypatch.setattr(cq, "_run_perf", lambda: {"api_health_ms": 0.5})  # way over band
    assert cq.main(["--perf"]) == 1
    assert json.loads(bl.read_text())["perf"]["api_health_ms"] == 0.2  # unchanged
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_check_quality.py::test_perf_ok_band -v`
Expected: FAIL — `perf_ok` not defined.

- [ ] **Step 3: Write minimal implementation**

Add to `scripts/check_quality.py` (near the top constants):

```python
PERF_TOLERANCE = 1.10
```

Add the helpers (below `ratchet`):

```python
def perf_ok(current: float, baseline: float, tol: float = PERF_TOLERANCE) -> bool:
    return current <= baseline * tol


def perf_ratchet(current: dict, baseline: dict, tol: float = PERF_TOLERANCE) -> tuple[bool, dict]:
    """Perf metrics ratchet DOWN with a +tol band. ok=False if ANY exceeds band.
    Unknown metrics (not in baseline) are accepted and seeded."""
    ok_all = True
    new = dict(baseline)
    for key, cur in current.items():
        base = baseline.get(key)
        if base is None:
            new[key] = cur            # seed a newly-tracked metric
            continue
        if not perf_ok(cur, base, tol):
            ok_all = False
        new[key] = min(cur, base)
    if not ok_all:
        return False, baseline
    return True, new


def _run_perf() -> dict:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "perf_probe", REPO_ROOT / "scripts" / "perf_probe.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.measure()
```

Add `--perf` to the arg parser in `main` and handle it BEFORE the test/coverage path:

```python
    parser.add_argument("--perf", action="store_true",
                        help="measure hot-path latency; fail on >10% regression, ratchet down")
```
then, right after parsing args:
```python
    if args.perf:
        metrics = _run_perf()
        baseline = _load_baseline()
        ok, new_perf = perf_ratchet(metrics, baseline.get("perf", {}))
        if not ok:
            print(f"PERF: FAIL — regression. baseline={baseline.get('perf', {})} current={metrics}")
            return 1
        if new_perf != baseline.get("perf"):
            baseline["perf"] = new_perf
            _save_baseline(baseline)
            print(f"PERF: PASS — perf baseline updated to {new_perf}")
        else:
            print(f"PERF: PASS — perf baseline held {new_perf}")
        return 0
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_check_quality.py -v`
Expected: PASS (all prior + 6 new).

- [ ] **Step 5: Seed the real perf block**

Run: `.venv/bin/python scripts/check_quality.py --perf`
Expected: `PERF: PASS — perf baseline updated to {...}` and `test-baseline.json` now has a `perf` block with the four measured medians. Report the numbers. (The deterministic keys `tests_passed`/`coverage_pct`/`ruff_violations`/`mypy_errors` must be unchanged.)

- [ ] **Step 6: Verify deterministic gate untouched + full suite green**

Run: `.venv/bin/python scripts/check_quality.py --check-only` → `QUALITY: PASS` (still only the 4 deterministic metrics; the added `perf` key must not break it — confirm `ratchet`/`_measure` ignore unknown baseline keys).
Run: `.venv/bin/python -m pytest -q` → green.

- [ ] **Step 7: Commit**

```bash
git add scripts/check_quality.py tests/test_check_quality.py test-baseline.json
git commit -m "feat(perf): --perf ratchet mode + seed perf baseline"
```

---

### Task 3: CI (non-blocking) + pre-commit manual hook + docs

**Files:**
- Modify: `.github/workflows/quality.yml` (add a non-blocking perf step)
- Modify: `.pre-commit-config.yaml` (add a manual-stage perf hook)
- Modify: `README.md` (document `--perf`)
- Test: `tests/test_precommit_config.py` (assert the manual perf hook exists)

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_precommit_config.py
def test_precommit_has_manual_perf_hook():
    text = pathlib.Path(".pre-commit-config.yaml").read_text()
    assert "check_quality.py --perf" in text
    assert "manual" in text  # perf hook is manual-stage, not the default commit gate
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_precommit_config.py::test_precommit_has_manual_perf_hook -v`
Expected: FAIL — no perf hook yet.

- [ ] **Step 3: Add the config + docs**

Append to `.pre-commit-config.yaml` under the existing `local` repo's `hooks:` list:

```yaml
      - id: perf-gate
        name: performance gate (hot-path latency, +10% band)
        entry: .venv/bin/python scripts/check_quality.py --perf
        language: system
        pass_filenames: false
        stages: [manual]
```

In `.github/workflows/quality.yml`, add a **non-blocking** perf step after the existing check:

```yaml
      - name: performance (informational, non-blocking)
        run: .venv/bin/python scripts/check_quality.py --perf || true
```

In `README.md`, under Quality gates, add:

```markdown
- Performance: `.venv/bin/python scripts/check_quality.py --perf` measures hot-path latency
  (`predict_proba_raw` + key endpoints) and fails on a >10% regression vs the `perf` block in
  `test-baseline.json` (ratchets down on improvement). Perf is environment-sensitive, so it runs
  on-demand and as a **non-blocking** CI step (not in the every-turn Stop hook); run it locally via
  `.venv/bin/pre-commit run perf-gate --hook-stage manual`.
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_precommit_config.py -v` → PASS.
Run: full suite `.venv/bin/python -m pytest -q` → green.

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/quality.yml .pre-commit-config.yaml README.md tests/test_precommit_config.py
git commit -m "feat(perf): non-blocking CI perf step + manual pre-commit perf hook + docs"
```

---

## Notes for the implementer

- **Interpreter:** always the venv. `_run_perf` loads `perf_probe` by absolute path (`REPO_ROOT/scripts/perf_probe.py`); `perf_probe` inserts the repo root on `sys.path` so `import app` works when run as a script.
- **Do not add `--perf` to `--check-only` or the Stop hook** — perf stays out of the deterministic every-turn gate (noise).
- **Do not hand-edit the `perf` block** — only `check_quality.py --perf` writes it.
- **The seeded perf numbers are machine-relative** — the local `--perf` run is authoritative; CI perf is informational only.
- Confirm the existing deterministic `ratchet`/`_measure`/`main --check-only` still pass with the new `perf` key present in `test-baseline.json` (they read only their four keys; the extra key is ignored).
