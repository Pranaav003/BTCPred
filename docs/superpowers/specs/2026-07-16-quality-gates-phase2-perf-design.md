# Quality Gates — Phase 2 (Performance Baselines) — Design

**Date:** 2026-07-16
**Status:** Approved for planning
**Author:** Claude + Pranaav

Phase 2 of the engineering-standards charter. Phase 1 (lint/type ratchet) is merged.

---

## 1. Purpose

Add **performance baselines** for the app's hot paths, recorded in `test-baseline.json`
under a `perf` key, gated at **+10% regression** and **ratcheting down** (faster = new
baseline). The perf gate runs **on-demand and in CI**, NOT in the every-turn Stop hook
(perf is environment-sensitive; the deterministic gate stays test/coverage/ruff/mypy only).

## 2. Measured baseline (this machine, venv, median of 100–200 runs)

| Hot path | Median | p95 |
|---|---|---|
| `predict_proba_raw` (model inference) | 43 ms | 66 ms |
| `GET /api/control/state` | 0.85 ms | 1.0 ms |
| `GET /api/settings` | 0.92 ms | 1.0 ms |
| `GET /api/health` | 0.14 ms | 0.15 ms |

## 3. Goals / Non-goals

### Goals
- A deterministic-as-possible perf probe (warmup + median of N runs) for the four hot paths.
- Store baselines in `test-baseline.json` under `perf` (median ms per path).
- `check_quality.py --perf`: measure, **fail if any path > baseline × 1.10**, ratchet the
  baseline **down** (min) on a faster run. Separate from `--check-only`.
- CI: a perf step that runs the probe and prints the numbers, **non-blocking** (see §5).
- Tests for the perf-ratchet logic (pure) via monkeypatch — never run the real probe in a
  unit test.

### Non-goals
- `compute_features` / full poll-cycle benchmarks (need candle/Kalshi fixtures) — deferred;
  the approved set is inference + the three endpoints.
- Perf in the every-turn Stop hook (explicitly excluded — noise).
- `pytest-benchmark` (its separate baseline store doesn't integrate with `test-baseline.json`;
  a lightweight custom probe keeps one source of truth).

## 4. Architecture

### 4.1 Perf probe
- `scripts/perf_probe.py` (or a function block in `check_quality.py`): `measure() -> dict`
  returning median ms per metric key: `predict_proba_raw_ms`, `api_control_state_ms`,
  `api_settings_ms`, `api_health_ms`. Each: warmup (~10) then median of N (~100–200) via
  `time.perf_counter`. Model inference uses a fixed feature dict; endpoints use a
  `create_app("testing")` test client.

### 4.2 Ratchet integration (`check_quality.py`)
- Reuse `ratchet_directional(current, baseline_val, "down")` from Phase 1 — with a **+10%
  tolerance**: a helper `perf_ok(current, baseline) -> bool` = `current <= baseline * 1.10`;
  on a passing run the stored baseline becomes `min(current, baseline)` (ratchet down).
- `--perf` mode: run `measure()`, load `baseline["perf"]`, fail (exit 1) if any metric
  violates the +10% band; else write the min-updated `perf` block. `--perf` is independent of
  the default (test/coverage/ruff/mypy) path so noise can't break the deterministic gate.
- `--init` seeds `perf` from a live `measure()`.

### 4.3 test-baseline.json shape (extended)
```json
{
  "tests_passed": 212, "coverage_pct": 40.0, "ruff_violations": 278, "mypy_errors": 38,
  "perf": {
    "predict_proba_raw_ms": 43.0,
    "api_control_state_ms": 0.9,
    "api_settings_ms": 0.9,
    "api_health_ms": 0.2
  }
}
```

### 4.4 Enforcement
- **Local / on-demand:** `.venv/bin/python scripts/check_quality.py --perf` (blocking).
- **CI:** a step running `--perf` but **non-blocking** (`|| true`, prints results) — see §5.
- **NOT** in the Stop hook or `--check-only`.

## 5. The CI-flakiness reality (honest treatment)

Microbenchmarks on shared CI runners are unreliable: the same 43 ms inference can measure
80–120 ms on a loaded GitHub runner, so a hard +10% gate against a locally-recorded baseline
would fail almost every CI run. Therefore:

- The **authoritative perf gate is the local `--perf` run** (consistent machine, where the
  baseline was recorded). Wire it into the `.pre-commit-config.yaml` as a **separate,
  optional manual hook** (run via `pre-commit run perf --hook-stage manual`), not the default
  commit gate.
- **CI perf is informational** — it prints the measured numbers for visibility but does
  **not fail the build** (`continue-on-error` / `|| true`). This is the honest limit of
  microbenchmark gating in shared CI; a hard CI perf gate would be flaky theater.

## 6. Testing

- Unit-test `perf_ok`: within band passes, above band fails, faster ratchets down (via direct
  calls — pure function, no probe).
- Unit-test `--perf` main path by monkeypatching `measure()` to return canned metrics:
  within band → exit 0 + baseline min-updated; over band → exit 1 + baseline unchanged.
- Never run the real probe inside a unit test (slow + noisy).
- Full suite stays green; seed the real `perf` block via `--init`/`--perf`.

## 7. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Perf noise breaks the deterministic gate | `--perf` is fully separate from `--check-only`/Stop hook |
| CI perf flakiness | CI perf is non-blocking/informational (§5) |
| Machine-specific baseline meaningless elsewhere | Authoritative gate is local; baseline documented as machine-relative |
| Model-load time skews first measurement | Warmup runs before timing |

## 8. Success criteria

- `scripts/check_quality.py --perf` measures the four paths, fails on a >10% regression,
  ratchets the baseline down on improvement; `test-baseline.json` has a seeded `perf` block.
- The deterministic `--check-only` gate is unchanged (still test/coverage/ruff/mypy only).
- CI prints perf numbers without failing the build; pre-commit exposes perf as a manual hook.
- Perf-ratchet logic is unit-tested via monkeypatch; full suite green.
