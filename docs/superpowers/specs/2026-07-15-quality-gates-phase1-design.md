# Quality Gates — Phase 1 (Lint/Type Ratchet + Baseline Consolidation) — Design

**Date:** 2026-07-15
**Status:** Approved for planning
**Author:** Claude + Pranaav

Phase 1 of the engineering-standards charter. Later phases (2 performance baselines,
3 error-handling audit, 4 coverage push, 5 full pre-commit gate) get their own specs.

---

## 1. Purpose

Establish **lint (ruff) and type (mypy) gates** for the project and fold them into the
existing ratcheting quality harness, so violations can only ever go **down** — the same
enforcement model already used for test count and coverage. Consolidate all quality metrics
into a single **`test-baseline.json`** (the charter's chosen filename). Add a real **git
pre-commit gate** so the checks run on commit, not only via the Claude Stop hook.

## 2. Background / current state

- The harness (merged) has: `pyproject.toml` (pytest+coverage), `TestingConfig`, conftest
  isolation, 204 tests, `scripts/check_quality.py` + `quality_baseline.json`
  (`tests_passed`, `coverage_pct`), a `.claude/settings.json` Stop hook, and GitHub CI —
  all running on a Python 3.13 `.venv`.
- **No ruff, mypy, pre-commit, or `test-baseline.json` exist yet.**
- Decisions locked with the user: keep the `tests/` dir (do not scatter tests into `app/`);
  adopt **ruff + mypy with ratchet-down** (not block-at-zero); performance baselines are
  Phase 2.

## 3. Goals / Non-goals

### Goals
- Add ruff + mypy (pinned) with config in `pyproject.toml`.
- Rename/migrate `quality_baseline.json` → **`test-baseline.json`** holding: `tests_passed`,
  `coverage_pct`, `ruff_violations`, `mypy_errors`.
- Extend `check_quality.py` to measure ruff + mypy counts and enforce a **directional
  ratchet**: `tests_passed`/`coverage_pct` may only rise; `ruff_violations`/`mypy_errors`
  may only fall. Any regression → exit 1.
- Update the Stop hook + CI to enforce the extended baseline (the hook command is unchanged;
  it already calls `check_quality.py --check-only`).
- Add `.pre-commit-config.yaml` running the ratchet on `git commit`; document `pre-commit install`.

### Non-goals (later phases)
- Performance baselines / `pytest-benchmark` (Phase 2).
- Error-handling audit + logging rotation (Phase 3).
- Raising coverage / adding feature tests (Phase 4) — Phase 1 must NOT fix lint/type
  violations wholesale; it only *measures and freezes* them as a ceiling.
- Moving tests adjacent to source (explicitly rejected).

## 4. Architecture

### 4.1 Tooling + config
- `requirements-dev.txt`: add `ruff==<pin>`, `mypy==<pin>` (implementer pins to the versions
  that install cleanly in the 3.13 venv).
- `pyproject.toml`:
  - `[tool.ruff]` — target py311, a sensible default rule set (`E`, `F`, `I`); `exclude`
    tests scratch and `.venv`. No auto-fixing in the gate.
  - `[tool.mypy]` — `python_version = "3.11"`, `ignore_missing_imports = true` (third-party
    stubs absent), non-strict (the codebase is largely untyped; strict would be noise).

### 4.2 `test-baseline.json` (migrated, single source of quality truth)
```json
{
  "tests_passed": <int>,
  "coverage_pct": <float>,
  "ruff_violations": <int>,
  "mypy_errors": <int>
}
```
`quality_baseline.json` is removed; every reference updates to `test-baseline.json`.

### 4.3 `scripts/check_quality.py` (extended)
- New pure helper `ratchet_directional(metric, current, baseline, direction) -> (ok, new)`:
  `direction="up"` → ok if `current >= baseline`, new = max; `direction="down"` → ok if
  `current <= baseline`, new = min. `--check-only` never moves the baseline.
- New `_run_ruff() -> int` (violation count) and `_run_mypy() -> int` (error count), each
  invoking the tool via the venv (`.venv/bin/ruff`, `.venv/bin/python -m mypy app sim`),
  parsing the count robustly (fall back to 0 / a large sentinel on parse failure so a
  broken parse can't silently pass).
- `main` composes all four metrics; **fails if ANY regresses**; on green (non-check-only),
  writes each metric in its improving direction. Prints a per-metric PASS/FAIL summary.
- Test-suite green remains a hard precondition (as today).

### 4.4 Enforcement points
- **Stop hook** (`.claude/settings.json`): command unchanged
  (`.venv/bin/python scripts/check_quality.py --check-only`) — now also enforces lint/type.
- **CI** (`.github/workflows/quality.yml`): already runs `check_quality.py --check-only`;
  ensure the workflow installs ruff+mypy (they're in `requirements-dev.txt`, already installed).
- **`.pre-commit-config.yaml`**: a `local` hook running
  `.venv/bin/python scripts/check_quality.py --check-only` on commit. Documented as opt-in
  via `pre-commit install` (pre-commit added to `requirements-dev.txt`).

## 5. Ratchet semantics (explicit)

| Metric | Direction | Regression (exit 1) | On green, baseline becomes |
|---|---|---|---|
| `tests_passed` | up | current < baseline | max(current, baseline) |
| `coverage_pct` | up | current < baseline | max(current, baseline) |
| `ruff_violations` | down | current > baseline | min(current, baseline) |
| `mypy_errors` | down | current > baseline | min(current, baseline) |

`--check-only` (hook/CI/pre-commit): fails on any regression, **never** rewrites the baseline.
Plain run (manual / Phase-end): applies all improving moves.

## 6. Testing

- Unit-test `ratchet_directional` for both directions: regression fails + holds baseline;
  improvement passes + moves baseline; equal passes + holds; `--check-only` never moves.
- Test that `check_quality` fails when a metric regresses and passes when all hold (via
  monkeypatched `_run_suite`/`_run_ruff`/`_run_mypy` — never run the real tools inside a test,
  to avoid recursion, per the existing pattern).
- Update `tests/test_check_quality.py` and `tests/test_hook_command_runs.py` for the new
  baseline filename + metrics.
- Seed the real `test-baseline.json` from a live run; full suite stays green.

## 7. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Ruff/mypy report huge counts, gate becomes noise | Ratchet-down freezes current counts as ceiling; no forced cleanup this phase |
| Parse failure silently passes gate | Parse failure yields a large sentinel (treated as regression), never 0 |
| Renaming baseline breaks hook/CI | Grep all refs; hook/CI reference the script, not the json; update README |
| mypy on untyped code is noisy | Non-strict config + `ignore_missing_imports`; counts frozen, not fixed |
| Stop hook now slower (adds ruff+mypy) | Acceptable; ruff is fast, mypy on app/sim is seconds |

## 8. Success criteria

- `ruff` + `mypy` installed and configured; `.venv/bin/ruff check` and `mypy app sim` run.
- `test-baseline.json` holds all four metrics, seeded from a real run; `quality_baseline.json`
  gone with no dangling references.
- `check_quality.py --check-only` fails if any metric regresses (test-verified via monkeypatch),
  passes when all hold; a plain run ratchets each in its improving direction.
- Stop hook, CI, and a new `.pre-commit-config.yaml` all enforce the extended baseline.
- Full test suite green; no lint/type violations were *fixed* this phase (measurement only).
