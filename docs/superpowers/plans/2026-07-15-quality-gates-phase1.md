# Quality Gates Phase 1 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add ruff + mypy gates to the quality harness with a directional ratchet (tests/coverage up, violations down), consolidate all metrics into `test-baseline.json`, and add a git pre-commit gate.

**Architecture:** Extend the existing `scripts/check_quality.py` + Stop-hook harness. New pure `ratchet_directional`, ruff/mypy count runners, a 4-metric baseline in `test-baseline.json` (migrated from `quality_baseline.json`), and a `.pre-commit-config.yaml`. Measure-and-freeze only — do NOT fix existing lint/type violations this phase.

**Tech Stack:** Python 3.13 venv, pytest, ruff, mypy, pre-commit.

## Global Constraints

- Use the venv for everything: **`.venv/bin/python -m pytest`**, `.venv/bin/ruff`, `.venv/bin/python -m mypy`, `.venv/bin/python scripts/check_quality.py`. Never bare `pytest`/`ruff`/`mypy` or `python3`.
- Do NOT put a module-level `from app import ...` in any test file (defeats conftest isolation) — use fixtures or lazy in-function imports.
- Ratchet directions: `tests_passed`↑, `coverage_pct`↑, `ruff_violations`↓, `mypy_errors`↓. Regression in ANY → exit 1. `--check-only` never moves the baseline.
- **Do NOT fix ruff/mypy violations this phase** — only measure and freeze the current counts as a ceiling.
- Tests that exercise `check_quality`'s CLI must monkeypatch the measurement functions — never run the real suite/ruff/mypy inside a test (recursion).
- Every task ends green (full suite) and commits. Branch: `feature/quality-gates-phase1`.

---

### Task 1: Directional ratchet pure function

**Files:**
- Modify: `scripts/check_quality.py` (add `ratchet_directional`, do not touch existing `ratchet`/`main` yet)
- Test: `tests/test_check_quality.py` (add cases; keep existing)

**Interfaces:**
- Produces: `ratchet_directional(current, baseline_val, direction) -> tuple[bool, number]` — `direction="up"`: ok if `current >= baseline_val`, moved = `max`; `direction="down"`: ok if `current <= baseline_val`, moved = `min`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_check_quality.py
def test_ratchet_directional_up():
    assert cq.ratchet_directional(205, 204, "up") == (True, 205)   # improve
    assert cq.ratchet_directional(204, 204, "up") == (True, 204)   # equal
    assert cq.ratchet_directional(203, 204, "up") == (False, 204)  # regress -> hold


def test_ratchet_directional_down():
    assert cq.ratchet_directional(5, 8, "down") == (True, 5)    # fewer violations = improve
    assert cq.ratchet_directional(8, 8, "down") == (True, 8)    # equal
    assert cq.ratchet_directional(9, 8, "down") == (False, 8)   # more = regress -> hold
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_check_quality.py::test_ratchet_directional_up -v`
Expected: FAIL — `AttributeError: module 'check_quality' has no attribute 'ratchet_directional'`.

- [ ] **Step 3: Write minimal implementation**

Add to `scripts/check_quality.py` (below the existing `ratchet` function):

```python
def ratchet_directional(current, baseline_val, direction: str):
    """(ok, moved). up: ok if current>=baseline, moved=max. down: ok if current<=baseline, moved=min."""
    if direction == "up":
        return (current >= baseline_val, max(current, baseline_val))
    return (current <= baseline_val, min(current, baseline_val))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_check_quality.py -v`
Expected: PASS (existing 5 + 2 new).

- [ ] **Step 5: Commit**

```bash
git add scripts/check_quality.py tests/test_check_quality.py
git commit -m "feat(quality): directional ratchet helper (up/down)"
```

---

### Task 2: ruff + mypy tooling, config, and count runners

**Files:**
- Modify: `requirements-dev.txt` (add ruff, mypy, pre-commit)
- Modify: `pyproject.toml` (add `[tool.ruff]`, `[tool.mypy]`)
- Modify: `scripts/check_quality.py` (add `_run_ruff`, `_run_mypy`, `VENV_RUFF`)
- Test: `tests/test_check_quality.py` (parser tests via monkeypatched subprocess)

**Interfaces:**
- Produces: `_run_ruff() -> int` (violation count; large sentinel `10**9` on parse failure), `_run_mypy() -> int` (error count; sentinel on parse failure).

- [ ] **Step 1: Install tools and pin versions**

Run: `.venv/bin/pip install ruff mypy pre-commit`
Then record the resolved versions into `requirements-dev.txt` (append, using the versions pip installed):

```
ruff==<resolved>
mypy==<resolved>
pre-commit==<resolved>
```

Run `.venv/bin/pip show ruff mypy pre-commit | grep -E "^(Name|Version)"` to get the exact versions to pin.

- [ ] **Step 2: Add tool config to `pyproject.toml`**

```toml
[tool.ruff]
target-version = "py311"
extend-exclude = [".venv", "sim_results"]

[tool.ruff.lint]
select = ["E", "F", "I"]

[tool.mypy]
python_version = "3.11"
ignore_missing_imports = true
follow_imports = "silent"
```

- [ ] **Step 3: Write the failing parser test**

```python
# append to tests/test_check_quality.py
import subprocess as _sp


def test_run_ruff_counts_found_errors(monkeypatch):
    def fake_run(*a, **k):
        return _sp.CompletedProcess(a, 1, stdout="app/x.py:1:1: F401 unused\nFound 3 errors.\n", stderr="")
    monkeypatch.setattr(cq.subprocess, "run", fake_run)
    assert cq._run_ruff() == 3


def test_run_ruff_zero_on_clean(monkeypatch):
    monkeypatch.setattr(cq.subprocess, "run",
                        lambda *a, **k: _sp.CompletedProcess(a, 0, stdout="All checks passed!\n", stderr=""))
    assert cq._run_ruff() == 0


def test_run_mypy_counts_found_errors(monkeypatch):
    def fake_run(*a, **k):
        return _sp.CompletedProcess(a, 1, stdout="app/x.py:1: error: bad\nFound 2 errors in 1 file\n", stderr="")
    monkeypatch.setattr(cq.subprocess, "run", fake_run)
    assert cq._run_mypy() == 2
```

- [ ] **Step 4: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_check_quality.py::test_run_ruff_counts_found_errors -v`
Expected: FAIL — `_run_ruff` not defined.

- [ ] **Step 5: Write minimal implementation**

Add to `scripts/check_quality.py` (near `VENV_PYTHON`):

```python
VENV_RUFF = REPO_ROOT / ".venv" / "bin" / "ruff"
_SENTINEL = 10**9  # parse-failure count: always treated as a regression, never a silent pass
```

And add the runners (below `_run_suite`):

```python
def _run_ruff() -> int:
    """Count ruff violations across app/sim/scripts. Sentinel on unparseable output."""
    proc = subprocess.run(
        [str(VENV_RUFF), "check", "app", "sim", "scripts", "--output-format=concise"],
        capture_output=True, text=True,
    )
    if proc.returncode == 0:
        return 0
    out = proc.stdout + proc.stderr
    m = re.search(r"Found (\d+) error", out)
    if m:
        return int(m.group(1))
    n = len(re.findall(r"^\S+:\d+:\d+:", out, re.M))
    return n if n > 0 else _SENTINEL


def _run_mypy() -> int:
    """Count mypy errors across app/sim. Sentinel on unparseable output."""
    proc = subprocess.run(
        [str(VENV_PYTHON), "-m", "mypy", "app", "sim"],
        capture_output=True, text=True,
    )
    if proc.returncode == 0:
        return 0
    out = proc.stdout + proc.stderr
    m = re.search(r"Found (\d+) error", out)
    if m:
        return int(m.group(1))
    n = len(re.findall(r": error:", out))
    return n if n > 0 else _SENTINEL
```

- [ ] **Step 6: Run tests + confirm the tools actually run**

Run: `.venv/bin/python -m pytest tests/test_check_quality.py -v`
Expected: PASS (existing + 3 new parser tests).

Run (informational — confirms the real tools execute and gives the counts you'll freeze next task):
`.venv/bin/ruff check app sim scripts --output-format=concise | tail -3` and
`.venv/bin/python -m mypy app sim | tail -3`
Expected: each prints a count (do NOT fix any violations — just observe).

- [ ] **Step 7: Commit**

```bash
git add requirements-dev.txt pyproject.toml scripts/check_quality.py tests/test_check_quality.py
git commit -m "feat(quality): add ruff+mypy config, deps, and count runners"
```

---

### Task 3: Consolidate into test-baseline.json + 4-metric ratchet

**Files:**
- Modify: `scripts/check_quality.py` (`BASELINE_PATH`, rewrite `ratchet`, add `_measure`, rewire `main`, `_load_baseline` defaults)
- Rename/migrate: `quality_baseline.json` → `test-baseline.json`
- Test: `tests/test_check_quality.py` (update old `ratchet`/CLI tests), `tests/test_hook_command_runs.py` (update monkeypatch target)

**Interfaces:**
- Consumes: `ratchet_directional`, `_run_ruff`, `_run_mypy`, `_run_suite`.
- Produces: `ratchet(current: dict, baseline: dict, allow_move: bool = True) -> tuple[bool, dict]`; `_measure() -> tuple[bool, dict]` (all_passed, metrics dict with the 4 keys); `BASELINE_PATH = REPO_ROOT / "test-baseline.json"`.

- [ ] **Step 1: Write the failing tests (new 4-metric ratchet)**

Replace the OLD `ratchet(...)` tests in `tests/test_check_quality.py` (the ones calling `cq.ratchet(105, 55.0, base)` etc.) with:

```python
_DIRS = {"tests_passed": "up", "coverage_pct": "up",
         "ruff_violations": "down", "mypy_errors": "down"}


def _base():
    return {"tests_passed": 204, "coverage_pct": 40.0, "ruff_violations": 50, "mypy_errors": 20}


def test_ratchet_all_hold_when_equal():
    ok, new = cq.ratchet(_base(), _base())
    assert ok is True and new == _base()


def test_ratchet_all_improves_every_metric():
    cur = {"tests_passed": 210, "coverage_pct": 45.0, "ruff_violations": 30, "mypy_errors": 10}
    ok, new = cq.ratchet(cur, _base())
    assert ok is True and new == cur  # all moved in the improving direction


def test_ratchet_fails_on_test_regression():
    cur = {"tests_passed": 203, "coverage_pct": 40.0, "ruff_violations": 50, "mypy_errors": 20}
    ok, new = cq.ratchet(cur, _base())
    assert ok is False and new == _base()


def test_ratchet_fails_on_more_ruff_violations():
    cur = {"tests_passed": 204, "coverage_pct": 40.0, "ruff_violations": 51, "mypy_errors": 20}
    ok, new = cq.ratchet(cur, _base())
    assert ok is False


def test_ratchet_check_only_never_moves():
    cur = {"tests_passed": 210, "coverage_pct": 45.0, "ruff_violations": 30, "mypy_errors": 10}
    ok, new = cq.ratchet(cur, _base(), allow_move=False)
    assert ok is True and new == _base()
```

Delete the now-obsolete old tests that call `ratchet` with the old `(passed, coverage_pct, baseline)` signature (`test_regression_in_tests_fails`, `test_coverage_regression_fails`, `test_improvement_ratchets_up`, `test_equal_is_ok_and_holds_baseline`, `test_check_only_never_raises_baseline`). Keep `test_ratchet_directional_*` and the ruff/mypy parser tests.

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_check_quality.py -v`
Expected: FAIL — `ratchet()` still has the old signature.

- [ ] **Step 3: Rewrite check_quality.py to the 4-metric model**

In `scripts/check_quality.py`:

Change the baseline path:
```python
BASELINE_PATH = REPO_ROOT / "test-baseline.json"
```

Add the directions map (near the top, after imports):
```python
_DIRECTIONS = {
    "tests_passed": "up",
    "coverage_pct": "up",
    "ruff_violations": "down",
    "mypy_errors": "down",
}
```

Replace the old `ratchet(passed, coverage_pct, baseline, allow_raise=True)` with:
```python
def ratchet(current: dict, baseline: dict, allow_move: bool = True) -> tuple[bool, dict]:
    """Apply the directional ratchet to all metrics. ok=False if ANY regresses; never moves the wrong way."""
    ok_all = True
    moved = dict(baseline)
    for metric, direction in _DIRECTIONS.items():
        ok, new_val = ratchet_directional(current[metric], baseline[metric], direction)
        ok_all = ok_all and ok
        moved[metric] = new_val
    if not ok_all:
        return False, baseline
    return True, (moved if allow_move else baseline)
```

Add `_measure` (below `_run_mypy`):
```python
def _measure() -> tuple[bool, dict]:
    all_passed, passed, coverage = _run_suite()
    metrics = {
        "tests_passed": passed,
        "coverage_pct": coverage,
        "ruff_violations": _run_ruff(),
        "mypy_errors": _run_mypy(),
    }
    return all_passed, metrics
```

Update `_load_baseline` default:
```python
def _load_baseline() -> dict:
    if BASELINE_PATH.exists():
        return json.loads(BASELINE_PATH.read_text())
    return {"tests_passed": 0, "coverage_pct": 0.0,
            "ruff_violations": _SENTINEL, "mypy_errors": _SENTINEL}
```

Rewrite `main`:
```python
def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true",
                        help="fail on regression but never move the baseline")
    parser.add_argument("--init", action="store_true",
                        help="seed the baseline from the current run")
    args = parser.parse_args(argv)

    all_passed, metrics = _measure()
    if not all_passed:
        print(f"QUALITY: FAIL — tests not green {metrics}")
        return 1

    if args.init or not BASELINE_PATH.exists():
        _save_baseline(metrics)
        print(f"QUALITY: baseline seeded {metrics}")
        return 0

    baseline = _load_baseline()
    ok, new = ratchet(metrics, baseline, allow_move=not args.check_only)
    if not ok:
        print(f"QUALITY: FAIL — regression. baseline={baseline} current={metrics}")
        return 1
    if new != baseline:
        _save_baseline(new)
        print(f"QUALITY: PASS — baseline updated to {new}")
    else:
        print(f"QUALITY: PASS — baseline held {baseline}")
    return 0
```

- [ ] **Step 4: Update the hook-command test's monkeypatch target**

In `tests/test_hook_command_runs.py`, the tests monkeypatch `cq._run_suite`. `main` now calls `_measure`. Update both to monkeypatch `_measure`:
```python
def test_check_only_returns_zero_when_suite_green(monkeypatch):
    cq = _load_cq()
    monkeypatch.setattr(cq, "_measure", lambda: (True, {
        "tests_passed": 10_000, "coverage_pct": 100.0,
        "ruff_violations": 0, "mypy_errors": 0}))
    assert cq.main(["--check-only"]) == 0


def test_check_only_returns_one_when_suite_red(monkeypatch):
    cq = _load_cq()
    # all_passed=False makes main return 1 early, so the metric values are irrelevant.
    monkeypatch.setattr(cq, "_measure", lambda: (False, {
        "tests_passed": 0, "coverage_pct": 0.0,
        "ruff_violations": 0, "mypy_errors": 0}))
    assert cq.main(["--check-only"]) == 1
```

- [ ] **Step 5: Migrate the baseline file and seed real counts**

```bash
git rm quality_baseline.json
```
Then seed the new file from a live run:
Run: `.venv/bin/python scripts/check_quality.py --init`
Expected: writes `test-baseline.json` with real `tests_passed` (~204), `coverage_pct` (~40.0), `ruff_violations` (observed count), `mypy_errors` (observed count). Report the seeded numbers.

- [ ] **Step 6: Verify green + no dangling refs**

Run: `.venv/bin/python -m pytest -q`  → all pass.
Run: `grep -rIn "quality_baseline.json" . --exclude-dir=.venv --exclude-dir=.git` → only historical references in `docs/` (plans/reports) are acceptable; NO reference in `scripts/`, `.claude/`, `.github/`, `README.md`. Fix any live reference to point at `test-baseline.json`.
Run: `.venv/bin/python scripts/check_quality.py --check-only` → `QUALITY: PASS`.

- [ ] **Step 7: Commit**

```bash
git add scripts/check_quality.py tests/test_check_quality.py tests/test_hook_command_runs.py test-baseline.json
git rm --cached quality_baseline.json 2>/dev/null || true
git commit -m "feat(quality): consolidate 4-metric directional ratchet into test-baseline.json"
```

---

### Task 4: Git pre-commit gate

**Files:**
- Create: `.pre-commit-config.yaml`
- Test: `tests/test_precommit_config.py`

**Interfaces:**
- Produces: a `.pre-commit-config.yaml` with a `local` hook running `.venv/bin/python scripts/check_quality.py --check-only`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_precommit_config.py
import pathlib


def test_precommit_config_runs_quality_gate():
    p = pathlib.Path(".pre-commit-config.yaml")
    assert p.exists()
    text = p.read_text()
    assert "repos:" in text                          # it is a pre-commit config
    assert "check_quality.py --check-only" in text   # and it runs our gate
```

Note: keep the test dependency-free (assert on file text), since PyYAML may not be installed.

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_precommit_config.py -v`
Expected: FAIL — file missing.

- [ ] **Step 3: Create the config**

```yaml
# .pre-commit-config.yaml
# Install once with: .venv/bin/pre-commit install
repos:
  - repo: local
    hooks:
      - id: quality-gate
        name: quality gate (tests + coverage + ruff + mypy ratchet)
        entry: .venv/bin/python scripts/check_quality.py --check-only
        language: system
        pass_filenames: false
        always_run: true
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_precommit_config.py -v`  → PASS.
Run full suite: `.venv/bin/python -m pytest -q` → green.

- [ ] **Step 5: Commit**

```bash
git add .pre-commit-config.yaml tests/test_precommit_config.py
git commit -m "feat(quality): git pre-commit gate runs the quality ratchet"
```

---

### Task 5: Docs + final baseline

**Files:**
- Modify: `README.md`
- Test: full suite + ratchet

- [ ] **Step 1: Update README quality-harness section**

Replace/extend the quality section so it documents the 4-metric gate:

```markdown
## Quality gates
- Setup: `python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt -r requirements-dev.txt`
- Run tests: `.venv/bin/python -m pytest`
- Lint: `.venv/bin/ruff check app sim scripts` · Types: `.venv/bin/python -m mypy app sim`
- Ratchet: `.venv/bin/python scripts/check_quality.py` enforces `test-baseline.json`:
  `tests_passed` and `coverage_pct` may only rise; `ruff_violations` and `mypy_errors` may only fall.
  `--check-only` fails on any regression without moving the baseline.
- Enforced by: the `.claude/settings.json` Stop hook, GitHub CI, and a git pre-commit hook
  (`.venv/bin/pre-commit install`).
```

- [ ] **Step 2: Verify + ratchet**

Run: `.venv/bin/python -m pytest -q` → green.
Run: `.venv/bin/python scripts/check_quality.py` → `QUALITY: PASS` (baseline holds or improves).

- [ ] **Step 3: Commit**

```bash
git add README.md test-baseline.json
git commit -m "docs(quality): document ruff/mypy ratchet + pre-commit gate"
```

---

## Notes for the implementer

- **Do not fix lint/type violations this phase.** If ruff/mypy report N violations, that N becomes the frozen ceiling in `test-baseline.json`. Fixing them is later-phase work; here we only prevent *new* ones.
- **Never run the real suite/ruff/mypy inside a test** — monkeypatch `_measure`/`_run_ruff`/`_run_mypy` (recursion + speed). Same pattern as the existing hook test.
- **Interpreter:** always the venv (`.venv/bin/...`). If `.venv` is missing: `python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt -r requirements-dev.txt`.
- **Baseline file is machine-managed** — only `check_quality.py` writes `test-baseline.json`; never hand-edit.
- If the seeded `ruff_violations`/`mypy_errors` are very large, that's fine and expected for an untyped/unlinted codebase — the point is they can only go down.
