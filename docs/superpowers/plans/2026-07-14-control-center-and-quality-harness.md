# Strategy Control Center + Quality Harness — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a real test/quality harness (pytest config, fixtures, E2E pipeline test, ratcheting coverage baseline, auto-run hook) and a paper-default Strategy Control Center page to BTCPred.

**Architecture:** Harness first (Tasks 1–8) so the Control Center (Tasks 9–14) is built test-driven under the ratchet. Keep the existing Flask + Jinja + vanilla-JS + custom-CSS stack; no build step. Existing Dashboard/Monitor/Analytics pages are untouched — the Control Center is a new, additive page.

**Tech Stack:** Python 3, Flask, SQLAlchemy, pytest + pytest-cov, Jinja2, vanilla JS, custom CSS.

## Global Constraints

- Run tests with the **`pytest` on PATH** (Python 3.9, which has `pytest` + `pytest-cov`). Do NOT use `python3 -m pytest` — the system `python3` (3.14) has no pytest. All `Run:` commands below use bare `pytest`.
- Keep the no-build stack: Jinja templates + vanilla JS + custom CSS. No framework, no npm.
- Settings are string key/values in the `AppSettings` table, read/written via `app.db_helpers.get_setting(key, default=None)` and `set_setting(key, value)`.
- The Control Center defaults to **PAPER**; live trading stays OFF by default and requires explicit action.
- Existing pages (`/dashboard`, `/monitor`, `/analytics`, `/settings`) must keep working — do not modify their templates/JS.
- Every task ends with a green focused test and a commit. Do not lower the coverage/test baseline.
- Branch: `feature/control-center-and-harness`.

---

### Task 1: TestingConfig + pytest/coverage config + dev deps

**Files:**
- Create: `pyproject.toml`
- Create: `requirements-dev.txt`
- Modify: `app/config.py` (add `TestingConfig`, register `"testing"` in `config_by_name`)
- Test: `tests/test_app_factory.py`

**Interfaces:**
- Consumes: `create_app(config_name)` from `app` (existing), `config_by_name` from `app.config`.
- Produces: `TestingConfig` (TESTING=True, `SQLALCHEMY_DATABASE_URI="sqlite:///:memory:"`); `config_by_name["testing"]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_app_factory.py
from app import create_app
from app.config import config_by_name


def test_testing_config_registered():
    assert "testing" in config_by_name


def test_create_app_testing_uses_memory_db_and_testing_flag():
    app = create_app("testing")
    assert app.testing is True
    assert app.config["SQLALCHEMY_DATABASE_URI"] == "sqlite:///:memory:"


def test_create_app_testing_does_not_start_scheduler():
    app = create_app("testing")
    assert getattr(app, "scheduler_instance", None) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_app_factory.py -v`
Expected: FAIL — `KeyError: 'testing'` in `create_app`.

- [ ] **Step 3: Write minimal implementation**

Add to `app/config.py` after `ProductionConfig` and before `config_by_name`:

```python
class TestingConfig(BaseConfig):
    """Testing settings: in-memory DB, no scheduler, CSRF off."""

    TESTING = True
    DEBUG = False
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
    SQLALCHEMY_ENGINE_OPTIONS = {}
```

And update the dict:

```python
config_by_name = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
}
```

Create `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "unit: pure unit tests",
    "integration: DB/API integration tests",
    "e2e: end-to-end pipeline tests",
]

[tool.coverage.run]
source = ["app", "sim"]
omit = ["tests/*", "*/__pycache__/*"]

[tool.coverage.report]
show_missing = true
```

Create `requirements-dev.txt`:

```
pytest==8.3.5
pytest-cov==7.1.0
coverage==7.10.7
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_app_factory.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml requirements-dev.txt app/config.py tests/test_app_factory.py
git commit -m "test: add TestingConfig, pytest/coverage config, dev deps"
```

---

### Task 2: Shared conftest fixtures

**Files:**
- Create: `tests/conftest.py`
- Test: `tests/test_conftest_fixtures.py`

**Interfaces:**
- Consumes: `create_app("testing")`.
- Produces: pytest fixtures `app` (app with pushed context, in-memory DB created), `client` (Flask test client). Both function-scoped so each test gets a fresh DB.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_conftest_fixtures.py
def test_client_fixture_hits_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.get_json()["status"] in ("ok", "healthy", "up")


def test_app_fixture_has_app_context(app):
    from app.db_helpers import get_setting
    # seeded default exists (seed_default_settings ran in create_app)
    assert get_setting("signal_mode") is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_conftest_fixtures.py -v`
Expected: FAIL — fixtures `client`/`app` not found.

- [ ] **Step 3: Write minimal implementation**

```python
# tests/conftest.py
"""Shared pytest fixtures for the BTCPred test suite."""
import pytest

from app import create_app


@pytest.fixture
def app():
    """A fresh testing app with an app context and in-memory DB."""
    application = create_app("testing")
    ctx = application.app_context()
    ctx.push()
    try:
        yield application
    finally:
        ctx.pop()


@pytest.fixture
def client(app):
    """Flask test client bound to the testing app."""
    return app.test_client()
```

Note: if `test_client_fixture_hits_health` fails on the exact `status` string, read `app/routes/api.py` `/api/health` handler and assert the actual value it returns.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_conftest_fixtures.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add tests/conftest.py tests/test_conftest_fixtures.py
git commit -m "test: shared conftest app/client fixtures on testing config"
```

---

### Task 3: Page-render smoke tests (existing pages)

**Files:**
- Test: `tests/test_pages_render.py`

**Interfaces:**
- Consumes: `client` fixture.
- Produces: nothing (characterization tests).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pages_render.py
import pytest


@pytest.mark.parametrize("path", ["/dashboard", "/monitor", "/analytics", "/settings"])
def test_page_renders_200(client, path):
    resp = client.get(path)
    assert resp.status_code == 200
    assert b"<html" in resp.data.lower()


def test_root_redirects(client):
    resp = client.get("/")
    assert resp.status_code in (301, 302)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pages_render.py -v`
Expected: PASS or FAIL depending on templates. If any page returns 500 (e.g. needs data), READ that route in `app/routes/dashboard.py` and either provide the minimal seeded data it needs or mark that single case `xfail(reason=...)`. Do not weaken the 200 assertion for pages that already render.

- [ ] **Step 3: Write minimal implementation**

No production code needed if pages render. If a page 500s only due to a missing model file, guard the test:

```python
# add near top of test file
import os
MODEL_PRESENT = os.path.exists("raw_feature_model.pkl")
```

and skip only the affected case with `@pytest.mark.skipif(not MODEL_PRESENT, reason="model artifact required")`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pages_render.py -v`
Expected: PASS (all parametrized + redirect).

- [ ] **Step 5: Commit**

```bash
git add tests/test_pages_render.py
git commit -m "test: page-render smoke tests for existing dashboard pages"
```

---

### Task 4: API settings round-trip integration test

**Files:**
- Test: `tests/test_api_settings.py`

**Interfaces:**
- Consumes: `client` fixture; `GET/POST /api/settings` (existing).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api_settings.py
def test_get_settings_returns_dict(client):
    resp = client.get("/api/settings", headers={"Accept": "application/json"})
    assert resp.status_code == 200
    assert isinstance(resp.get_json(), dict)


def test_post_settings_roundtrip(client):
    resp = client.post(
        "/api/settings",
        json={"mispricing_threshold": 0.25},
        headers={"Accept": "application/json"},
    )
    assert resp.status_code == 200
    from app.db_helpers import get_setting
    assert float(get_setting("mispricing_threshold")) == 0.25
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_api_settings.py -v`
Expected: PASS if endpoint behaves; if the POST shape differs, READ `POST /api/settings` in `app/routes/api.py` and match its accepted payload/keys exactly. Do not change the endpoint.

- [ ] **Step 3: Write minimal implementation**

No production change (characterization). If the endpoint requires a different content type, adjust the test to match the real handler.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_api_settings.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add tests/test_api_settings.py
git commit -m "test: /api/settings GET/POST round-trip integration"
```

---

### Task 5: End-to-end pipeline test

**Files:**
- Test: `tests/test_e2e_pipeline.py`

**Interfaces:**
- Consumes: `app` fixture; `app.model_loader.predict_proba_raw(feature_dict) -> float`; `app.signal_engine.evaluate_ensemble_signal(...)`; `app.paper_trading.execute_paper_trade(...)`; `app.resolver` resolution + `app.db_helpers` PnL; `app.models.PaperTrade`.

**Note to implementer:** This is an integration test against existing code. Before writing, READ the exact signatures of `evaluate_ensemble_signal`, `execute_paper_trade`, and the resolver's paper-resolution/PnL function. The code below is the intended shape; adapt argument names to the real signatures. If a signature is fundamentally incompatible with a deterministic no-network test, report BLOCKED with the specific signature rather than guessing.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_e2e_pipeline.py
import os
import pytest

MODEL_PRESENT = os.path.exists("raw_feature_model.pkl")


@pytest.mark.e2e
@pytest.mark.skipif(not MODEL_PRESENT, reason="model artifact required")
def test_model_inference_returns_probability():
    from app.model_loader import predict_proba_raw
    # Minimal feature dict; missing features default to 0.0 inside predict_proba_raw.
    p = predict_proba_raw({"seconds_to_close": 100, "return_1m": 5.0})
    assert 0.0 <= p <= 1.0


@pytest.mark.e2e
def test_paper_trade_executes_and_resolves(app):
    """DB-backed pipeline slice: place a paper trade, resolve it, assert PnL sign."""
    from app.models import PaperTrade
    from app.extensions import db
    from app.paper_trading import execute_paper_trade

    # Execute a paper BUY NO on a synthetic market at p_market=0.40 (NO entry 0.60).
    # Adapt kwargs to the real execute_paper_trade signature.
    trade = execute_paper_trade(
        ticker="KXBTC15M-TEST0000",
        side="no",
        contracts=1,
        entry_price=0.60,
        seconds_to_close=100,
        signal_triggered=True,
    )
    assert trade is not None
    stored = PaperTrade.query.filter_by(ticker="KXBTC15M-TEST0000").first()
    assert stored is not None
    assert stored.resolved is False

    # Resolve as NO wins (event did not happen). Set outcome + call the resolver
    # PnL path. Adapt to the real resolver function name.
    from app.db_helpers import resolve_paper_trade_outcome  # adapt name if different
    resolve_paper_trade_outcome(stored, final_outcome_yes=0)
    db.session.refresh(stored)
    assert stored.resolved is True
    assert stored.realized_pnl > 0  # NO won -> positive PnL
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_e2e_pipeline.py -v`
Expected: FAIL — until signatures are matched (import error or arg mismatch).

- [ ] **Step 3: Write minimal implementation**

No new production code — this test exercises existing functions. Fix the test by matching real signatures discovered by reading `app/paper_trading.py`, `app/resolver.py`, `app/db_helpers.py`. If the resolver only resolves via market lookup, insert a `Market` row with `resolved=True, final_outcome_yes=0` and call the real `resolve_pending_markets()`; assert the trade's `realized_pnl > 0`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_e2e_pipeline.py -v`
Expected: PASS (model test may skip if artifact absent; the DB pipeline test must pass).

- [ ] **Step 5: Commit**

```bash
git add tests/test_e2e_pipeline.py
git commit -m "test: end-to-end pipeline (model inference + paper trade + resolution)"
```

---

### Task 6: Ratcheting quality baseline

**Files:**
- Create: `scripts/check_quality.py`
- Create: `quality_baseline.json`
- Test: `tests/test_check_quality.py`

**Interfaces:**
- Produces: `check_quality.ratchet(passed, coverage_pct, baseline, allow_raise=True) -> tuple[bool, dict]` (pure function); a CLI `python scripts/check_quality.py [--check-only] [--init]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_check_quality.py
import importlib.util
import pathlib

_spec = importlib.util.spec_from_file_location(
    "check_quality", pathlib.Path("scripts/check_quality.py")
)
cq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cq)


def test_regression_in_tests_fails():
    base = {"tests_passed": 100, "coverage_pct": 50.0}
    ok, new = cq.ratchet(99, 50.0, base)
    assert ok is False
    assert new == base  # unchanged on failure


def test_coverage_regression_fails():
    base = {"tests_passed": 100, "coverage_pct": 50.0}
    ok, new = cq.ratchet(100, 49.9, base)
    assert ok is False


def test_improvement_ratchets_up():
    base = {"tests_passed": 100, "coverage_pct": 50.0}
    ok, new = cq.ratchet(105, 55.0, base)
    assert ok is True
    assert new == {"tests_passed": 105, "coverage_pct": 55.0}


def test_equal_is_ok_and_holds_baseline():
    base = {"tests_passed": 100, "coverage_pct": 50.0}
    ok, new = cq.ratchet(100, 50.0, base)
    assert ok is True
    assert new == base


def test_check_only_never_raises_baseline():
    base = {"tests_passed": 100, "coverage_pct": 50.0}
    ok, new = cq.ratchet(105, 55.0, base, allow_raise=False)
    assert ok is True
    assert new == base  # not raised in check-only mode
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_check_quality.py -v`
Expected: FAIL — `scripts/check_quality.py` does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/check_quality.py
"""Run the suite with coverage; fail on regression; ratchet the baseline up on success."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

BASELINE_PATH = Path(__file__).resolve().parent.parent / "quality_baseline.json"


def ratchet(passed: int, coverage_pct: float, baseline: dict,
            allow_raise: bool = True) -> tuple[bool, dict]:
    """Return (ok, new_baseline). ok=False on any regression. Never lowers."""
    if passed < baseline["tests_passed"] or coverage_pct < baseline["coverage_pct"]:
        return False, baseline
    if not allow_raise:
        return True, baseline
    new = {
        "tests_passed": max(passed, baseline["tests_passed"]),
        "coverage_pct": max(coverage_pct, baseline["coverage_pct"]),
    }
    return True, new


def _run_suite() -> tuple[bool, int, float]:
    """Run pytest with coverage. Returns (all_passed, passed_count, coverage_pct)."""
    proc = subprocess.run(
        ["pytest", "--cov=app", "--cov=sim", "--cov-report=term-missing", "-q"],
        capture_output=True, text=True,
    )
    out = proc.stdout + proc.stderr
    # passed count
    m_pass = re.search(r"(\d+) passed", out)
    passed = int(m_pass.group(1)) if m_pass else 0
    # coverage total line: "TOTAL   ...   NN%"
    m_cov = re.search(r"TOTAL\s+.*?(\d+(?:\.\d+)?)%", out)
    coverage = float(m_cov.group(1)) if m_cov else 0.0
    all_passed = proc.returncode == 0
    return all_passed, passed, coverage


def _load_baseline() -> dict:
    if BASELINE_PATH.exists():
        return json.loads(BASELINE_PATH.read_text())
    return {"tests_passed": 0, "coverage_pct": 0.0}


def _save_baseline(baseline: dict) -> None:
    BASELINE_PATH.write_text(json.dumps(baseline, indent=2) + "\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true",
                        help="fail on regression but never raise the baseline")
    parser.add_argument("--init", action="store_true",
                        help="seed the baseline from the current run")
    args = parser.parse_args(argv)

    all_passed, passed, coverage = _run_suite()
    if not all_passed:
        print(f"QUALITY: FAIL — tests not green (passed={passed}, cov={coverage}%)")
        return 1

    if args.init or not BASELINE_PATH.exists():
        _save_baseline({"tests_passed": passed, "coverage_pct": coverage})
        print(f"QUALITY: baseline seeded (tests={passed}, cov={coverage}%)")
        return 0

    baseline = _load_baseline()
    ok, new = ratchet(passed, coverage, baseline, allow_raise=not args.check_only)
    if not ok:
        print(f"QUALITY: FAIL — regression vs baseline {baseline} "
              f"(got tests={passed}, cov={coverage}%)")
        return 1
    if new != baseline:
        _save_baseline(new)
        print(f"QUALITY: PASS — baseline ratcheted up to {new}")
    else:
        print(f"QUALITY: PASS — baseline held at {baseline}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Create `quality_baseline.json` with a conservative floor (it will ratchet up on first real run):

```json
{
  "tests_passed": 0,
  "coverage_pct": 0.0
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_check_quality.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Seed the real baseline and commit**

Run: `python3 scripts/check_quality.py --init` (this invokes `pytest` internally via PATH)
Expected: prints `QUALITY: baseline seeded (tests=NN, cov=XX%)` and writes real numbers into `quality_baseline.json`.

```bash
git add scripts/check_quality.py quality_baseline.json tests/test_check_quality.py
git commit -m "test: ratcheting quality baseline runner + pure ratchet logic tests"
```

---

### Task 7: Auto-run Stop hook

**Files:**
- Create: `.claude/settings.json` (or modify if it exists)
- Test: `tests/test_hook_command_runs.py`

**Interfaces:**
- Consumes: `scripts/check_quality.py`.
- Produces: a `Stop` hook that runs `python3 scripts/check_quality.py --check-only`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_hook_command_runs.py
import importlib.util
import json
import pathlib


def _load_cq():
    spec = importlib.util.spec_from_file_location(
        "check_quality", pathlib.Path("scripts/check_quality.py"))
    cq = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cq)
    return cq


def test_settings_json_has_stop_hook():
    p = pathlib.Path(".claude/settings.json")
    assert p.exists()
    cfg = json.loads(p.read_text())
    hooks = cfg.get("hooks", {})
    assert "Stop" in hooks
    blob = json.dumps(hooks["Stop"])
    assert "check_quality.py" in blob


def test_check_only_returns_zero_when_suite_green(monkeypatch):
    # Validate the hook command's LOGIC without actually running pytest again.
    # (Running the real suite here would recurse: the subprocess pytest would
    # re-collect this test, spawning pytest endlessly.)
    cq = _load_cq()
    monkeypatch.setattr(cq, "_run_suite", lambda: (True, 10_000, 100.0))
    assert cq.main(["--check-only"]) == 0


def test_check_only_returns_one_when_suite_red(monkeypatch):
    cq = _load_cq()
    monkeypatch.setattr(cq, "_run_suite", lambda: (False, 0, 0.0))
    assert cq.main(["--check-only"]) == 1
```

Note: we deliberately do NOT run the real suite inside a test — the hook itself (`check_quality.py --check-only`, invoked by the Stop hook at turn end, not inside pytest) exercises the real run. Monkeypatching `_run_suite` tests the CLI/exit-code logic safely.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_hook_command_runs.py::test_settings_json_has_stop_hook -v`
Expected: FAIL — `.claude/settings.json` missing or no Stop hook.

- [ ] **Step 3: Write minimal implementation**

Create `.claude/settings.json` (merge into existing `hooks` if the file already exists — do not clobber other keys):

```json
{
  "hooks": {
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "python3 scripts/check_quality.py --check-only"
          }
        ]
      }
    ]
  }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_hook_command_runs.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add .claude/settings.json tests/test_hook_command_runs.py
git commit -m "chore: Stop hook runs ratcheting quality check every turn"
```

---

### Task 8 (OPTIONAL): GitHub Actions CI

Skip this task if the user declined CI. Otherwise:

**Files:**
- Create: `.github/workflows/quality.yml`

- [ ] **Step 1: Create the workflow**

```yaml
# .github/workflows/quality.yml
name: quality
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt -r requirements-dev.txt
      - run: pytest --cov=app --cov=sim -q
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/quality.yml
git commit -m "ci: run pytest + coverage on push"
```

---

### Task 9: Paper-by-default seed settings

**Files:**
- Modify: `app/db_helpers.py` (`seed_default_settings`)
- Test: `tests/test_paper_default_seed.py`

**Interfaces:**
- Consumes: `app` fixture, `get_setting`.
- Produces: fresh apps come up with `paper_trading_enabled="true"`, `auto_trade_enabled="true"`, `live_trading_enabled="false"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_paper_default_seed.py
def test_paper_defaults_on_live_off(app):
    from app.db_helpers import get_setting
    assert get_setting("paper_trading_enabled") == "true"
    assert get_setting("auto_trade_enabled") == "true"
    assert get_setting("live_trading_enabled") == "false"
    assert get_setting("mispricing_threshold") == "0.25"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_paper_default_seed.py -v`
Expected: FAIL — `paper_trading_enabled`/`auto_trade_enabled` seed to `"false"` today.

- [ ] **Step 3: Write minimal implementation**

In `app/db_helpers.py`, inside the `seed_default_settings()` defaults dict, change these two values from `"false"` to `"true"`:

```python
        "auto_trade_enabled": "true",
        "paper_trading_enabled": "true",
```

Leave `live_trading_enabled` at `"false"` and `mispricing_threshold` at `"0.25"` (already correct).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_paper_default_seed.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add app/db_helpers.py tests/test_paper_default_seed.py
git commit -m "feat: default fresh deploys to paper-trading the validated strategy"
```

---

### Task 10: GET /api/control/state endpoint

**Files:**
- Modify: `app/routes/api.py` (add endpoint)
- Test: `tests/test_control_api.py`

**Interfaces:**
- Consumes: `client` fixture; existing `get_setting`.
- Produces: `GET /api/control/state` → JSON with keys: `mode` (`"paper"`|`"live"`), `scheduler_running` (bool), `paper_trading_enabled` (bool), `auto_trade_enabled` (bool), `signal_mode` (str), `mispricing_threshold` (float), `breakeven_win_rate` (float), `trades_today` (int), `paper_pnl_today` (float).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_control_api.py
def test_control_state_shape(client):
    resp = client.get("/api/control/state", headers={"Accept": "application/json"})
    assert resp.status_code == 200
    data = resp.get_json()
    for key in ("mode", "scheduler_running", "paper_trading_enabled",
                "auto_trade_enabled", "signal_mode", "mispricing_threshold",
                "breakeven_win_rate", "trades_today", "paper_pnl_today"):
        assert key in data
    assert data["mode"] == "paper"  # default seed = live off
    assert data["paper_trading_enabled"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_control_api.py::test_control_state_shape -v`
Expected: FAIL — 404 (endpoint missing).

- [ ] **Step 3: Write minimal implementation**

Add to `app/routes/api.py` (near the other `@api_bp.route` handlers; the blueprint prefix is `/api`):

```python
@api_bp.route("/control/state", methods=["GET"])
def control_state():
    from app.db_helpers import get_setting
    live_on = get_setting("live_trading_enabled", "false") == "true"

    # paper P&L today + trade count today (reuse existing helper if present)
    trades_today, pnl_today = 0, 0.0
    try:
        from app.paper_trading import get_realized_pnl_today_utc
        pnl_today = float(get_realized_pnl_today_utc() or 0.0)
    except Exception:
        pnl_today = 0.0
    try:
        from app.models import PaperTrade
        from datetime import datetime, timezone
        start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        trades_today = PaperTrade.query.filter(PaperTrade.entry_at >= start).count()
    except Exception:
        trades_today = 0

    return jsonify({
        "mode": "live" if live_on else "paper",
        "scheduler_running": get_setting("scheduler_running", "false") == "true",
        "paper_trading_enabled": get_setting("paper_trading_enabled", "false") == "true",
        "auto_trade_enabled": get_setting("auto_trade_enabled", "false") == "true",
        "signal_mode": get_setting("signal_mode", "ensemble"),
        "mispricing_threshold": float(get_setting("mispricing_threshold", "0.25")),
        "breakeven_win_rate": 0.67,
        "trades_today": trades_today,
        "paper_pnl_today": pnl_today,
    })
```

Note: `jsonify` is already imported in `api.py`. If `get_realized_pnl_today_utc` has a different name, read `app/paper_trading.py` and use the real one; the `try/except` keeps the endpoint safe if absent.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_control_api.py::test_control_state_shape -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/routes/api.py tests/test_control_api.py
git commit -m "feat: GET /api/control/state aggregate for the control center"
```

---

### Task 11: POST /api/control/apply-defaults endpoint

**Files:**
- Modify: `app/routes/api.py` (add endpoint)
- Test: `tests/test_control_api.py` (extend)

**Interfaces:**
- Consumes: `client` fixture; `apply_backtest_settings.VALIDATED_SETTINGS`; `set_setting`.
- Produces: `POST /api/control/apply-defaults` → applies validated paper config; returns `{"updated": [...]}`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_control_api.py
def test_apply_defaults_sets_validated_paper_config(client):
    resp = client.post("/api/control/apply-defaults",
                       headers={"Accept": "application/json"})
    assert resp.status_code == 200
    assert "updated" in resp.get_json()
    from app.db_helpers import get_setting
    assert get_setting("signal_mode") == "ensemble"
    assert float(get_setting("mispricing_threshold")) == 0.25
    assert get_setting("paper_trading_enabled") == "true"
    assert get_setting("live_trading_enabled") == "false"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_control_api.py::test_apply_defaults_sets_validated_paper_config -v`
Expected: FAIL — 404.

- [ ] **Step 3: Write minimal implementation**

Add to `app/routes/api.py`:

```python
@api_bp.route("/control/apply-defaults", methods=["POST"])
def control_apply_defaults():
    from app.db_helpers import set_setting
    from apply_backtest_settings import VALIDATED_SETTINGS

    updated = []
    # Normalize threshold string "0.2500" -> "0.25" for display consistency.
    values = dict(VALIDATED_SETTINGS)
    values["mispricing_threshold"] = "0.25"
    # Paper on, live off.
    values.update({
        "signal_mode": "ensemble",
        "paper_trading_enabled": "true",
        "auto_trade_enabled": "true",
        "live_trading_enabled": "false",
    })
    for key, val in values.items():
        set_setting(key, val)
        updated.append(key)
    return jsonify({"updated": updated})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_control_api.py -v`
Expected: PASS (both control API tests).

- [ ] **Step 5: Commit**

```bash
git add app/routes/api.py tests/test_control_api.py
git commit -m "feat: POST /api/control/apply-defaults applies validated paper config"
```

---

### Task 12: /control route + redirect + nav + template

**Files:**
- Modify: `app/routes/dashboard.py` (add `/control` route; change `/` to redirect to `/control`)
- Modify: `app/templates/base.html` (add nav link to Control)
- Create: `app/templates/control.html`
- Test: `tests/test_pages_render.py` (extend)

**Interfaces:**
- Consumes: `client` fixture.
- Produces: `/control` renders `control.html` (contains "Strategy Control Center"); `/` redirects to `/control`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_pages_render.py
def test_control_page_renders(client):
    resp = client.get("/control")
    assert resp.status_code == 200
    assert b"Strategy Control Center" in resp.data


def test_root_redirects_to_control(client):
    resp = client.get("/")
    assert resp.status_code in (301, 302)
    assert "/control" in resp.headers.get("Location", "")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pages_render.py::test_control_page_renders -v`
Expected: FAIL — 404 for `/control`.

- [ ] **Step 3: Write minimal implementation**

In `app/routes/dashboard.py`, change the existing `/` handler to redirect to `/control` (it currently redirects to `/dashboard`) and add a `/control` route:

```python
@dashboard_bp.route("/control")
def control():
    return render_template("control.html")
```

Ensure the `/` route redirects to `url_for("dashboard.control")` (update the existing redirect target).

In `app/templates/base.html`, add a nav link alongside the existing sidebar links (match the existing `<a>` markup pattern):

```html
<a href="{{ url_for('dashboard.control') }}" class="nav-link">Control</a>
```

Create `app/templates/control.html`:

```html
{% extends "base.html" %}
{% block content %}
<div class="control-page">
  <header class="control-head">
    <h1>Strategy Control Center</h1>
    <span id="mode-badge" class="mode-badge">PAPER MODE</span>
  </header>

  <section class="control-card" id="mode-card">
    <h2>Mode</h2>
    <div class="mode-toggle">
      <button id="mode-paper" class="mode-btn active">● PAPER (default)</button>
      <button id="mode-live" class="mode-btn danger">LIVE — real money</button>
    </div>
    <p class="hint" id="mode-hint">Live is OFF · safe. Switching to LIVE requires typed confirmation.</p>
  </section>

  <section class="control-card" id="status-card">
    <h2>Status</h2>
    <ul class="status-list">
      <li>Scheduler: <span id="st-scheduler">—</span></li>
      <li>Trades today: <span id="st-trades">—</span></li>
      <li>Paper P&amp;L today: <span id="st-pnl">—</span></li>
    </ul>
  </section>

  <section class="control-card" id="strategy-card">
    <div class="row-between">
      <h2>Strategy (validated)</h2>
      <button id="apply-defaults-btn" class="btn primary">Apply validated defaults</button>
    </div>
    <label>Signal mode
      <select id="signal-mode-select">
        <option value="ensemble">Ensemble</option>
        <option value="agreement">Agreement</option>
        <option value="mispricing">Mispricing</option>
      </select>
    </label>
    <label>Mispricing gap threshold
      <input id="threshold-input" type="number" step="0.01" min="0" max="0.6">
    </label>
    <button id="save-btn" class="btn">Save</button>
  </section>
</div>
<script defer src="{{ url_for('static', filename='js/control.js') }}"></script>
{% endblock %}
```

Also add to `base.html` `<head>` (or the block that loads page CSS) a link to `control.css` — follow how `settings.css` is loaded and mirror it for the control page. If `base.html` uses a `{% block head %}`, put the stylesheet link in `control.html`'s head block instead.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pages_render.py -v`
Expected: PASS (existing + control render + redirect).

- [ ] **Step 5: Commit**

```bash
git add app/routes/dashboard.py app/templates/base.html app/templates/control.html tests/test_pages_render.py
git commit -m "feat: /control page (Strategy Control Center) + default landing"
```

---

### Task 13: control.css + control.js

**Files:**
- Create: `app/static/css/control.css`
- Create: `app/static/js/control.js`
- Test: `tests/test_control_static.py`

**Interfaces:**
- Consumes: `GET /api/control/state`, `POST /api/control/apply-defaults`, `GET/POST /api/settings`, `GET/POST /api/scheduler/*` (existing). Browser behavior is not unit-tested (Python-level test depth); we assert the static files exist and the template references them, and rely on the already-tested API contracts.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_control_static.py
import pathlib


def test_control_static_files_exist():
    assert pathlib.Path("app/static/js/control.js").exists()
    assert pathlib.Path("app/static/css/control.css").exists()


def test_control_template_references_assets():
    html = pathlib.Path("app/templates/control.html").read_text()
    assert "js/control.js" in html


def test_control_js_calls_state_and_defaults_endpoints():
    js = pathlib.Path("app/static/js/control.js").read_text()
    assert "/api/control/state" in js
    assert "/api/control/apply-defaults" in js
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_control_static.py -v`
Expected: FAIL — files/asset refs missing.

- [ ] **Step 3: Write minimal implementation**

Create `app/static/css/control.css`:

```css
.control-page { max-width: 760px; margin: 0 auto; padding: 24px; }
.control-head { display: flex; justify-content: space-between; align-items: center; }
.mode-badge { padding: 6px 12px; border-radius: 999px; background: var(--success); color: #04120a; font-weight: 600; }
.mode-badge.live { background: var(--danger); color: #1a0505; }
.control-card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 20px; margin-top: 16px; }
.mode-toggle { display: flex; gap: 12px; }
.mode-btn { flex: 1; padding: 14px; border-radius: 10px; border: 1px solid var(--border); background: transparent; color: var(--text-primary); cursor: pointer; }
.mode-btn.active { border-color: var(--success); }
.mode-btn.danger { color: var(--danger); }
.status-list { list-style: none; padding: 0; display: grid; gap: 8px; }
.row-between { display: flex; justify-content: space-between; align-items: center; }
.hint { color: var(--text-secondary); font-size: 13px; }
.btn { padding: 10px 16px; border-radius: 8px; border: 1px solid var(--border); background: transparent; color: var(--text-primary); cursor: pointer; }
.btn.primary { background: var(--accent); border-color: var(--accent); color: white; }
label { display: block; margin: 12px 0; }
```

Create `app/static/js/control.js`:

```javascript
// Strategy Control Center — focused page logic (no dependency on main.js).
async function loadState() {
  const res = await fetch("/api/control/state", { headers: { Accept: "application/json" } });
  if (!res.ok) return;
  const s = await res.json();
  const badge = document.getElementById("mode-badge");
  badge.textContent = s.mode === "live" ? "LIVE — REAL MONEY" : "PAPER MODE";
  badge.classList.toggle("live", s.mode === "live");
  document.getElementById("st-scheduler").textContent = s.scheduler_running ? "RUNNING" : "STOPPED";
  document.getElementById("st-trades").textContent = s.trades_today;
  document.getElementById("st-pnl").textContent = "$" + Number(s.paper_pnl_today).toFixed(2);
  const sel = document.getElementById("signal-mode-select");
  if (sel) sel.value = s.signal_mode;
  const thr = document.getElementById("threshold-input");
  if (thr) thr.value = s.mispricing_threshold;
}

async function applyDefaults() {
  await fetch("/api/control/apply-defaults", { method: "POST", headers: { Accept: "application/json" } });
  await loadState();
}

async function saveStrategy() {
  const payload = {
    signal_mode: document.getElementById("signal-mode-select").value,
    mispricing_threshold: Number(document.getElementById("threshold-input").value),
  };
  await fetch("/api/settings", {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify(payload),
  });
  await loadState();
}

async function setLive(enabled) {
  if (enabled) {
    const typed = window.prompt('Type "LIVE" to enable REAL-MONEY trading:');
    if (typed !== "LIVE") return;
  }
  await fetch("/api/settings", {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify({ live_trading_enabled: enabled }),
  });
  await loadState();
}

document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("apply-defaults-btn")?.addEventListener("click", applyDefaults);
  document.getElementById("save-btn")?.addEventListener("click", saveStrategy);
  document.getElementById("mode-paper")?.addEventListener("click", () => setLive(false));
  document.getElementById("mode-live")?.addEventListener("click", () => setLive(true));
  loadState();
  setInterval(loadState, 5000);
});
```

Ensure `control.html` loads `control.css` (add `<link rel="stylesheet" href="{{ url_for('static', filename='css/control.css') }}">` in the appropriate head block, mirroring how `settings.html` loads `settings.css`).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_control_static.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add app/static/css/control.css app/static/js/control.js app/templates/control.html tests/test_control_static.py
git commit -m "feat: control center CSS + JS (status strip, apply-defaults, paper/live toggle)"
```

---

### Task 14: Docs + final full-suite green

**Files:**
- Modify: `README.md` (or create `docs/CONTROL_CENTER.md` if README is large)
- Test: full suite + baseline

**Interfaces:** none.

- [ ] **Step 1: Update docs**

Add a section to `README.md` (create the file if absent) covering:

```markdown
## Quality harness
- Run tests: `pytest` (uses `pyproject.toml`; Python 3.9 `pytest` on PATH).
- Coverage + ratchet: `python3 scripts/check_quality.py` (raises the baseline in
  `quality_baseline.json` on success; `--check-only` fails on regression without raising).
- A `.claude/settings.json` Stop hook runs `check_quality.py --check-only` after each change.

## Strategy Control Center
- `/control` is the default landing page. It defaults to PAPER; LIVE requires typed confirmation.
- "Apply validated defaults" sets the validated ensemble config (threshold 0.25, moderate profile,
  NO gate 0.20, entry caps) and turns paper trading on.
- Fresh deploys come up paper-trading the validated strategy; live trading stays OFF until enabled.
```

- [ ] **Step 2: Run the full suite + ratchet**

Run: `pytest -q`
Expected: all green.

Run: `python3 scripts/check_quality.py`
Expected: `QUALITY: PASS — baseline ratcheted up to {...}` (final test count + coverage).

- [ ] **Step 3: Commit**

```bash
git add README.md quality_baseline.json
git commit -m "docs: quality harness + control center usage; final baseline"
```

---

## Notes for the implementer

- **Interpreter:** always invoke `pytest` (PATH, Python 3.9). `python3 script.py` is fine for scripts, but tests run under the PATH `pytest`.
- **Do not modify** existing page templates/JS (`main.js`, `monitor.js`, `analytics.js`, their templates) beyond the `base.html` nav link and the `/` redirect target.
- **E2E test (Task 5)** is the one place you must read real signatures before writing; escalate BLOCKED if a deterministic no-network test isn't achievable, rather than guessing.
- **Ratchet during the build:** as each task adds tests, the Stop hook (Task 7 onward) ratchets the baseline up automatically; never edit `quality_baseline.json` by hand except via `check_quality.py`.
- **JS is not unit-tested** (Python-level test depth by design); its correctness rests on the tested API contracts (`/api/control/state`, `/api/control/apply-defaults`, `/api/settings`) plus manual check.
