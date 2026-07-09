# BTCPred Live-Loss Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the live account's structural bleed and restore profitability by fixing the self-reverting threshold guard, gating NO trades to the model's calibrated zone, capping interim position size, and adding a retrain-validation gate.

**Architecture:** Phase 0 (Tasks 1–4) ships code + config guards immediately: fix the seed guard that reverts `mispricing_threshold`, add a tunable `no_max_p_raw` NO gate in the signal engine, add a tunable `live_max_contracts` clamp in the scheduler, and push validated settings via the deploy script. Phase 1 (Task 5) adds `validate_retrain.py`, the out-of-sample gate for a retrained model. Phase 2 is an operational step (clear the interim cap) captured in the runbook.

**Tech Stack:** Python 3.14, Flask, SQLAlchemy, scikit-learn (RandomForest), pandas/numpy, pytest. Settings are key-value rows in the `app_settings` table read via `app.db_helpers.get_setting`/`set_setting`.

## Global Constraints

- Settings are strings in `AppSettings`; read with `get_setting(key, default)`, write with `set_setting(key, value)`. Empty string / `None` means "unset".
- New guards MUST be tunable AppSettings, not hardcoded constants (matches existing pattern).
- Signal-engine functions (`evaluate_ensemble_signal`, `evaluate_mispricing_signal`) are pure — no DB/app-context access inside them. Settings are read only in `evaluate_live_signal` and threaded in as parameters.
- `mispricing_threshold` production target = `0.25`; `no_max_p_raw` default = `0.20`; interim `live_max_contracts` = `1`.
- Keep existing backstops intact: `$4` trade-size cap, `max_risk = balance × 0.10`, `max_daily_loss`.
- Do not reduce existing test coverage; `pytest tests/` must stay green.
- Run tests with `python3 -m pytest`.

---

### Task 1: Fix the self-reverting `mispricing_threshold` seed guard + seed new keys

**Problem being fixed:** `seed_default_settings()` runs on every startup and resets `mispricing_threshold` to `0.10` whenever it exceeds `0.20`. This silently reverts the validated `0.25` on every deploy — the mechanism behind the config drift. Also seed the two new guard keys.

**Files:**
- Modify: `app/db_helpers.py:253-292` (defaults dict + the correction block)
- Test: `tests/test_seed_settings_threshold.py` (create)

**Interfaces:**
- Consumes: `get_setting(key, default=None) -> str | None`, `set_setting(key, value) -> AppSettings` (existing, `app/db_helpers.py`).
- Produces: `seed_default_settings()` behavior — after seeding, `mispricing_threshold` default is `"0.25"`; a stored value is only force-corrected when `> 0.60` (absurd, would never fire); new keys `no_max_p_raw="0.20"` and `live_max_contracts=""` are seeded when missing.

- [ ] **Step 1: Write the failing test**

Create `tests/test_seed_settings_threshold.py`:

```python
"""seed_default_settings must NOT revert a validated 0.25 mispricing_threshold,
and must seed the new guard keys. Regression for the config-drift mechanism."""
import importlib.util
import os
import sys
import types


def _load_db_helpers_with_fake_store():
    """Load app.db_helpers with get_setting/set_setting backed by an in-memory dict."""
    store = {}

    def fake_get(key, default=None):
        return store.get(key, default)

    def fake_set(key, value):
        store[key] = value
        return None

    # Stub the 'app' package and app.models so db_helpers imports cleanly.
    app_stub = types.ModuleType("app")
    app_stub.__path__ = [os.path.join(os.path.dirname(__file__), "..", "app")]
    sys.modules["app"] = app_stub
    models_stub = types.ModuleType("app.models")
    for name in ("AppSettings", "Market", "PaperTrade", "Portfolio", "Signal", "db"):
        setattr(models_stub, name, type(name, (), {}))
    sys.modules["app.models"] = models_stub
    sys.modules.setdefault("sqlalchemy", types.ModuleType("sqlalchemy"))
    sys.modules["sqlalchemy"].func = type("func", (), {})
    orm = types.ModuleType("sqlalchemy.orm")
    orm.contains_eager = lambda *a, **k: None
    sys.modules["sqlalchemy.orm"] = orm

    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app", "db_helpers.py"))
    spec = importlib.util.spec_from_file_location("app.db_helpers", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["app.db_helpers"] = mod
    spec.loader.exec_module(mod)
    # Swap in fakes for the store-backed helpers used by seed_default_settings.
    mod.get_setting = fake_get
    mod.set_setting = fake_set
    return mod, store


def test_seed_does_not_revert_025_threshold():
    mod, store = _load_db_helpers_with_fake_store()
    store["mispricing_threshold"] = "0.25"
    mod.seed_default_settings()
    assert store["mispricing_threshold"] == "0.25"  # NOT reverted to 0.10


def test_seed_default_threshold_is_025_when_missing():
    mod, store = _load_db_helpers_with_fake_store()
    mod.seed_default_settings()
    assert store["mispricing_threshold"] == "0.25"


def test_seed_corrects_absurd_threshold():
    mod, store = _load_db_helpers_with_fake_store()
    store["mispricing_threshold"] = "0.90"
    mod.seed_default_settings()
    assert store["mispricing_threshold"] == "0.25"


def test_seed_new_guard_keys():
    mod, store = _load_db_helpers_with_fake_store()
    mod.seed_default_settings()
    assert store["no_max_p_raw"] == "0.20"
    assert store["live_max_contracts"] == ""
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m pytest tests/test_seed_settings_threshold.py -v`
Expected: FAIL — `test_seed_does_not_revert_025_threshold` asserts `0.25` but current code reverts to `0.10`; new-key tests KeyError/fail.

- [ ] **Step 3: Implement the fix**

In `app/db_helpers.py`, in the `defaults` dict inside `seed_default_settings()` change the `mispricing_threshold` line and add two keys:

```python
        "mispricing_threshold": "0.25",
        "max_entry_price_yes": "0.65",
        "max_entry_price_no": "0.80",
        "no_max_p_raw": "0.20",
        "live_max_contracts": "",
```

Then replace the downward-correction block (currently reverts when `> 0.20`) with an absurd-only guard:

```python
    # Only correct a truly absurd threshold that would never fire (>0.60).
    # 0.25 is the backtest-validated production value and MUST survive restart.
    current_threshold = get_setting("mispricing_threshold")
    if current_threshold and float(current_threshold) > 0.60:
        set_setting("mispricing_threshold", "0.25")
        logger.info(
            "Corrected absurd mispricing_threshold from %s to 0.25", current_threshold
        )
```

Note: `""` is falsy, so the existing `if get_setting(key) is None` seeding guard still seeds `live_max_contracts` (its value is only skipped when the key already exists). Because `get_setting` returns the default `None` only when the row is missing, seeding `""` works; subsequent runs see the stored `""` and skip.

- [ ] **Step 4: Run the test to verify it passes**

Run: `python3 -m pytest tests/test_seed_settings_threshold.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Run the full suite to check for regressions**

Run: `python3 -m pytest tests/ -q`
Expected: all pass (no test asserted the old revert-to-0.10 behavior; if one does, update it to expect 0.25 and note it in the commit).

- [ ] **Step 6: Commit**

```bash
git add app/db_helpers.py tests/test_seed_settings_threshold.py
git commit -m "Fix seed guard that reverted mispricing_threshold to 0.10

The startup seed reset any threshold >0.20 to 0.10, silently undoing the
validated 0.25 on every deploy — the mechanism behind the live config drift.
Default is now 0.25; only absurd (>0.60) values are corrected. Also seed
no_max_p_raw and live_max_contracts guard keys."
```

---

### Task 2: Add tunable `no_max_p_raw` NO gate to the signal engine

**Problem being fixed:** the model's NO edge is real only where `p_raw < 0.20`; in `p_raw` 0.2–0.5 the market out-forecasts the model and NO bets are negative-EV. Block NO signals unless `p_raw < no_max_p_raw`.

**Files:**
- Modify: `app/signal_engine.py` — `evaluate_ensemble_signal` (add param + gate), `evaluate_mispricing_signal` (add param + gate), `evaluate_live_signal` (read setting, pass to both).
- Test: `tests/test_no_max_p_raw_gate.py` (create)

**Interfaces:**
- Consumes: existing `evaluate_ensemble_signal(...)` and `evaluate_mispricing_signal(...)` signatures; `get_setting` in `evaluate_live_signal`.
- Produces:
  - `evaluate_ensemble_signal(..., no_max_p_raw: float = 0.20)` — returns `signal == "NO SIGNAL"` for a would-be bearish NO when `p_raw >= no_max_p_raw`.
  - `evaluate_mispricing_signal(..., no_max_p_raw: float = 0.20)` — same gate on its NO path.

- [ ] **Step 1: Write the failing test**

Create `tests/test_no_max_p_raw_gate.py`:

```python
"""NO trades must be gated to the model's calibrated zone: p_raw < no_max_p_raw."""
import importlib.util
import os
import sys
import types


def _load_signal_engine():
    app_stub = types.ModuleType("app")
    app_stub.__path__ = [os.path.join(os.path.dirname(__file__), "..", "app")]
    app_stub.__package__ = "app"
    sys.modules["app"] = app_stub
    for name in ("flask", "flask_sqlalchemy", "flask_migrate", "flask_wtf", "dotenv", "click"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sqla = types.ModuleType("sqlalchemy"); sqla.func = type("func", (), {})
    sys.modules.setdefault("sqlalchemy", sqla)
    ml = types.ModuleType("app.model_loader"); ml.predict_proba_raw = lambda _: 0.5
    sys.modules["app.model_loader"] = ml
    dbh = types.ModuleType("app.db_helpers")
    dbh.get_setting = lambda k, d=None: d
    dbh.set_setting = lambda k, v: None
    sys.modules["app.db_helpers"] = dbh
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app", "signal_engine.py"))
    spec = importlib.util.spec_from_file_location("app.signal_engine", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["app.signal_engine"] = mod
    spec.loader.exec_module(mod)
    return mod


SE = _load_signal_engine()


def _ensemble_no(p_raw, no_max_p_raw=0.20):
    # market 0.62, model p_raw -> bearish gap >= 0.25 threshold; in-window; NO entry ok.
    return SE.evaluate_ensemble_signal(
        p_market=0.62, p_raw=p_raw, seconds_to_close=100, entry_bucket=60,
        yes_cutoff=0.72, max_entry_yes=0.65, max_entry_no=0.80,
        mispricing_threshold=0.25, min_seconds=60, max_seconds=120,
        no_max_p_raw=no_max_p_raw,
    )


def test_ensemble_blocks_no_when_praw_at_or_above_cap():
    # gap = 0.62 - 0.36 = 0.26 >= 0.25, but p_raw 0.36 >= 0.20 -> blocked.
    result = _ensemble_no(p_raw=0.36)
    assert result.signal == "NO SIGNAL"


def test_ensemble_allows_no_when_praw_below_cap():
    # gap = 0.62 - 0.15 = 0.47 >= 0.25, p_raw 0.15 < 0.20 -> NO allowed.
    result = _ensemble_no(p_raw=0.15)
    assert result.signal == "PAPER BUY NO"


def test_ensemble_gate_respects_setting_value():
    # With a looser cap of 0.40, p_raw 0.36 should be allowed again.
    result = _ensemble_no(p_raw=0.36, no_max_p_raw=0.40)
    assert result.signal == "PAPER BUY NO"


def test_mispricing_blocks_no_when_praw_above_cap():
    result = SE.evaluate_mispricing_signal(
        p_market=0.62, p_raw=0.36, seconds_to_close=100, entry_bucket=60,
        min_seconds=60, max_seconds=120, mispricing_threshold=0.25,
        max_entry_price_yes=0.65, max_entry_price_no=0.80, no_max_p_raw=0.20,
    )
    assert result.signal == "NO SIGNAL"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m pytest tests/test_no_max_p_raw_gate.py -v`
Expected: FAIL — `no_max_p_raw` is not a parameter yet (TypeError), and the gate is not implemented.

- [ ] **Step 3: Implement the gate in `evaluate_ensemble_signal`**

Add the parameter to the signature (after `volatility_guard_active`):

```python
    volatility_guard_active: bool = False,
    no_max_p_raw: float = 0.20,
) -> SignalResult:
```

After the line `mispricing_bearish = (-gap) >= thresh and float(p_raw) < 0.50`, add:

```python
    no_praw_ok = float(p_raw) < float(no_max_p_raw)
```

Change the bearish branch condition from:

```python
    elif mispricing_bearish and no_entry_ok:
```
to:
```python
    elif mispricing_bearish and no_entry_ok and no_praw_ok:
```

In the `else:` branch that builds `parts`, after the `if mispricing_bearish and not no_entry_ok:` block, add:

```python
        if mispricing_bearish and no_entry_ok and not no_praw_ok:
            parts.append(
                f"NO blocked: model p_raw {float(p_raw):.1%} >= no_max_p_raw "
                f"{float(no_max_p_raw):.1%} (edge only reliable below this)"
            )
```

- [ ] **Step 4: Implement the gate in `evaluate_mispricing_signal`**

Add the parameter to the signature (after `max_entry_price_no`):

```python
    max_entry_price_no: float = 1.0,
    no_max_p_raw: float = 0.20,
) -> SignalResult:
```

In the `elif result.signal == "PAPER BUY NO":` block, add this as the FIRST check (before the `no_price` min/max filters):

```python
    elif result.signal == "PAPER BUY NO":
        if float(p_raw) >= float(no_max_p_raw):
            result.signal = "NO SIGNAL"
            result.reason = (
                f"NO blocked: p_raw {float(p_raw):.1%} >= no_max_p_raw "
                f"{float(no_max_p_raw):.1%} (edge only reliable below this)"
            )
            result.entry_filtered = True
            return result
        no_price = 1.0 - float(p_market)
```

(The existing `no_price = 1.0 - float(p_market)` line that opened the block is now inside the guard's `else` flow — keep exactly one `no_price = ...` assignment before the min/max checks.)

- [ ] **Step 5: Thread the setting through `evaluate_live_signal`**

After the existing `max_entry_no = float(get_setting("max_entry_price_no", "0.80") or 0.80)` line, add:

```python
    no_max_p_raw = float(get_setting("no_max_p_raw", "0.20") or 0.20)
```

In the `signal_mode == "mispricing"` call to `evaluate_mispricing_signal(...)`, add:

```python
            max_entry_price_no=max_entry_no,
            no_max_p_raw=no_max_p_raw,
        )
```

In the `elif signal_mode == "ensemble"` call to `evaluate_ensemble_signal(...)`, add:

```python
            volatility_guard_active=volatility_guard_active,
            no_max_p_raw=no_max_p_raw,
        )
```

- [ ] **Step 6: Run the new test + full suite**

Run: `python3 -m pytest tests/test_no_max_p_raw_gate.py tests/test_signal_engine.py -v`
Expected: new tests PASS; existing `test_signal_engine.py` still PASS (defaults preserve prior behavior for p_raw already `< 0.20` cases; if any existing ensemble-NO test used `p_raw >= 0.20`, update it to pass `no_max_p_raw=1.0` and note it).

- [ ] **Step 7: Commit**

```bash
git add app/signal_engine.py tests/test_no_max_p_raw_gate.py
git commit -m "Gate NO trades to calibrated zone (p_raw < no_max_p_raw, default 0.20)

RCA showed the model's NO edge is real only when p_raw < 0.20; in 0.2-0.5
the market out-forecasts it and NO bets are negative-EV. New tunable
no_max_p_raw setting blocks NO signals above the cap in both ensemble and
mispricing modes."
```

---

### Task 3: Add tunable `live_max_contracts` clamp to the scheduler

**Problem being fixed:** interim safety — force ≤1 contract per live trade while the model is retrained, so residual mistakes cost cents. Implemented as a pure, unit-testable helper.

**Files:**
- Modify: `app/scheduler.py` — add `_apply_contract_cap` helper near `_aggressive_entry_price` (~line 170); call it in `_execute_live_trade` after `contracts = int(trade_size / entry_price)` (~line 407).
- Test: `tests/test_live_max_contracts.py` (create)

**Interfaces:**
- Produces: `_apply_contract_cap(contracts: int, live_max_contracts) -> int` — returns `min(contracts, cap)` when `live_max_contracts` parses to a positive int; returns `contracts` unchanged for `""`, `None`, non-numeric, or `<= 0`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_live_max_contracts.py`:

```python
"""_apply_contract_cap clamps live position size to the live_max_contracts setting."""
import os
import sys
import types

import pytest

_REAL_APP_DIR = os.path.join(os.path.dirname(__file__), "..", "app")
_SHADOW_ROOTS = ("app", "flask", "flask_sqlalchemy", "flask_migrate", "flask_wtf",
                 "sqlalchemy", "click", "dotenv")


def _is_shadow(key):
    return any(key == r or key.startswith(r + ".") for r in _SHADOW_ROOTS)


@pytest.fixture
def scheduler():
    saved = {k: sys.modules.pop(k) for k in list(sys.modules) if _is_shadow(k)}
    app_pkg = types.ModuleType("app"); app_pkg.__path__ = [_REAL_APP_DIR]; app_pkg.__package__ = "app"
    sys.modules["app"] = app_pkg
    try:
        import app.scheduler as scheduler
        yield scheduler
    finally:
        for k in [k for k in list(sys.modules) if _is_shadow(k)]:
            sys.modules.pop(k, None)
        sys.modules.update(saved)


def test_clamps_to_cap(scheduler):
    assert scheduler._apply_contract_cap(7, "1") == 1
    assert scheduler._apply_contract_cap(7, "3") == 3


def test_no_clamp_when_under_cap(scheduler):
    assert scheduler._apply_contract_cap(2, "5") == 2


def test_unset_means_no_cap(scheduler):
    assert scheduler._apply_contract_cap(9, "") == 9
    assert scheduler._apply_contract_cap(9, None) == 9


def test_nonnumeric_or_nonpositive_means_no_cap(scheduler):
    assert scheduler._apply_contract_cap(9, "abc") == 9
    assert scheduler._apply_contract_cap(9, "0") == 9
    assert scheduler._apply_contract_cap(9, "-2") == 9
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m pytest tests/test_live_max_contracts.py -v`
Expected: FAIL — `_apply_contract_cap` does not exist (AttributeError).

- [ ] **Step 3: Implement the helper**

In `app/scheduler.py`, immediately after the `_aggressive_entry_price` function (before `_execute_live_trade`), add:

```python
def _apply_contract_cap(contracts: int, live_max_contracts) -> int:
    """Clamp contract count to live_max_contracts when it is a positive int.

    Empty string, None, non-numeric, or <=0 all mean "no cap". Used for the
    interim tiny-fixed-size posture during model retraining.
    """
    try:
        cap = int(str(live_max_contracts).strip())
    except (TypeError, ValueError):
        return contracts
    if cap <= 0:
        return contracts
    return min(int(contracts), cap)
```

- [ ] **Step 4: Wire it into `_execute_live_trade`**

In `_execute_live_trade`, after the `if contracts < 1:` early-return block and before `actual_cost = contracts * entry_price`, insert:

```python
            live_max_contracts = get_setting("live_max_contracts", "")
            capped = _apply_contract_cap(contracts, live_max_contracts)
            if capped != contracts:
                logger.info(
                    "Contract cap: %s -> %s (live_max_contracts=%s)",
                    contracts, capped, live_max_contracts,
                )
                contracts = capped

            # price_cents (crossing the live ask) was computed above by
            # _aggressive_entry_price; entry_price is the expected fill price.
            actual_cost = contracts * entry_price
```

(Replace the existing single `actual_cost = contracts * entry_price` line — do not duplicate it.)

- [ ] **Step 5: Run the new test + full suite**

Run: `python3 -m pytest tests/test_live_max_contracts.py tests/test_aggressive_entry_price.py -q && python3 -m pytest tests/ -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add app/scheduler.py tests/test_live_max_contracts.py
git commit -m "Add live_max_contracts clamp for interim tiny-fixed-size posture

Pure _apply_contract_cap helper clamps live contract count to the
live_max_contracts setting (set to 1 during retraining). Empty/invalid/<=0
means no cap, so Phase 2 re-enables full sizing by clearing the setting."
```

---

### Task 4: Push validated settings via the deploy script

**Problem being fixed:** the validated settings must actually land in the production DB. Extend the one-shot deploy script to include the new keys and the corrected threshold.

**Files:**
- Modify: `apply_backtest_settings.py` — `VALIDATED_SETTINGS` dict.

**Interfaces:**
- Consumes: `get_setting`/`set_setting` (existing). No new code interfaces.
- Produces: running `python3 apply_backtest_settings.py` sets `mispricing_threshold=0.25`, `no_max_p_raw=0.20`, `live_max_contracts=1` (interim), plus the existing validated keys.

- [ ] **Step 1: Add the new keys to `VALIDATED_SETTINGS`**

In `apply_backtest_settings.py`, add these entries to the `VALIDATED_SETTINGS` dict (keep existing entries):

```python
    # NO-side calibration gate: model's NO edge is only reliable below this p_raw.
    "no_max_p_raw": "0.20",

    # Interim safety: hard 1-contract cap during model retraining (Phase 2 clears this).
    "live_max_contracts": "1",
```

Confirm the existing `"mispricing_threshold": "0.2500"` entry is present (it is).

- [ ] **Step 2: Sanity-check the script parses**

Run: `python3 -c "import ast; ast.parse(open('apply_backtest_settings.py').read()); print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add apply_backtest_settings.py
git commit -m "Add no_max_p_raw and interim live_max_contracts to deploy settings"
```

---

### Task 5: Add `validate_retrain.py` — the out-of-sample model gate

**Problem being fixed:** before re-enabling full sizing on a retrained model, prove it (a) beats the old model's Brier and (b) shows positive current-rule EV — on a held-out slice it never trained on, using its OWN re-predicted probabilities.

**Files:**
- Create: `validate_retrain.py`

**Interfaces:**
- Consumes: `raw_feature_model.pkl` (new bundle, on disk), a prior model bundle path for comparison, and the exported `live_training_data.csv`. Reuses the fill-cost/decision logic mirrored from `backtest_current_rule.py` (THRESH=0.25).
- Produces: prints a Brier comparison and current-rule EV on the held-out split, then `GATE: PASS` or `GATE: FAIL`. Exit code 0 on PASS, 1 on FAIL.

- [ ] **Step 1: Create the script**

Create `validate_retrain.py`:

```python
#!/usr/bin/env python3
"""Out-of-sample validation gate for a retrained model.

Splits the exported live data by market (most-recent ~20% held out, matching
merge_and_retrain.py), RE-PREDICTS p_raw with the NEW model on the held-out
rows' raw features (never reuses the logged p_raw column), and checks:
  (a) new-model Brier < old-model Brier on the held-out set, AND
  (b) current-rule (THRESH=0.25) replay EV/contract > 0 on the held-out set.
Exit 0 = PASS (safe to deploy + re-enable sizing), exit 1 = FAIL.

Usage:
  python3 validate_retrain.py --new raw_feature_model.pkl --old raw_feature_model.prev.pkl \
      --data live_training_data.csv
"""
from __future__ import annotations

import argparse
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

warnings.filterwarnings("ignore", category=FutureWarning)

THRESH = 0.25
MAX_YES, MAX_NO, MIN_ENTRY = 0.65, 0.80, 0.05
WIN_LO, WIN_HI = 90, 300
NO_MAX_P_RAW = 0.20
TEST_SIZE = 0.20


def entry_cost(side: str, price_now: float) -> float:
    if side == "YES":
        return min(0.99, price_now + 0.02)
    no_price = 1.0 - price_now
    off = 0.05 if no_price <= 0.40 else 0.03
    return min(0.99, no_price + off)


def decide(p_raw: float, price_now: float):
    gap = p_raw - price_now
    if gap >= THRESH and p_raw >= 0.50 and MIN_ENTRY <= price_now <= MAX_YES:
        return "YES"
    if (-gap) >= THRESH and p_raw < NO_MAX_P_RAW and MIN_ENTRY <= (1 - price_now) <= MAX_NO:
        return "NO"
    return None


def predict(bundle, df):
    model = bundle["model"]
    feats = bundle["features"]
    x = df.reindex(columns=feats).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return model.predict_proba(x)[:, 1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", default="raw_feature_model.pkl")
    ap.add_argument("--old", default="raw_feature_model.prev.pkl")
    ap.add_argument("--data", default="live_training_data.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    df["close_ts"] = pd.to_numeric(df["close_ts"], errors="coerce")
    df = df.dropna(subset=["close_ts", "price_now", "final_outcome_yes"])

    # Market-level temporal split: most-recent 20% of markets held out.
    order = df.groupby("market_ticker")["close_ts"].min().sort_values().index.tolist()
    n_test = max(1, int(len(order) * TEST_SIZE))
    test_tickers = set(order[len(order) - n_test:])
    test = df[df["market_ticker"].isin(test_tickers)].copy()
    print(f"held-out markets: {len(test_tickers)} | rows: {len(test)}")

    new_bundle = joblib.load(args.new)
    test["p_new"] = predict(new_bundle, test)
    y = test["final_outcome_yes"].astype(int).values
    new_brier = brier_score_loss(y, test["p_new"].values)

    old_brier = None
    try:
        old_bundle = joblib.load(args.old)
        old_brier = brier_score_loss(y, predict(old_bundle, test))
    except Exception as exc:
        print(f"WARN: could not load old model for comparison: {exc}")

    print(f"new Brier: {new_brier:.4f}" + (f" | old Brier: {old_brier:.4f}" if old_brier is not None else ""))

    # Current-rule replay on held-out set, one trade per market, NEW predictions.
    w = test[(test["seconds_to_close"] >= WIN_LO) & (test["seconds_to_close"] <= WIN_HI)].copy()
    w["decision"] = [decide(p, m) for p, m in zip(w["p_new"], pd.to_numeric(w["price_now"], errors="coerce"))]
    sig = w[w["decision"].notna()].sort_values("seconds_to_close", ascending=False)
    trades = sig.groupby("market_ticker", as_index=False).first()

    def pnl(r):
        won = (r.final_outcome_yes == 1) if r.decision == "YES" else (r.final_outcome_yes == 0)
        return (1.0 if won else 0.0) - entry_cost(r.decision, float(r.price_now))

    n = len(trades)
    if n:
        pnls = trades.apply(pnl, axis=1).values
        ev, wr, total = pnls.mean(), 100 * (pnls > 0).mean(), pnls.sum()
        se = pnls.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
        print(f"replay: n={n} EV/contract={ev:+.4f} WR={wr:.1f}% total={total:+.2f} 95%CI[{ev-1.96*se:+.4f},{ev+1.96*se:+.4f}]")
    else:
        ev = 0.0
        print("replay: n=0 (no qualifying trades in held-out window)")

    brier_ok = old_brier is None or new_brier < old_brier
    ev_ok = n > 0 and ev > 0
    if brier_ok and ev_ok:
        print("GATE: PASS")
        return 0
    print(f"GATE: FAIL (brier_ok={brier_ok}, ev_ok={ev_ok})")
    return 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Smoke-test the script parses and runs its help**

Run: `python3 validate_retrain.py --help`
Expected: argparse usage text (no import errors).

- [ ] **Step 3: Dry-run against the existing historical CSV (sanity only)**

Run: `python3 validate_retrain.py --new raw_feature_model.pkl --old raw_feature_model.pkl --data live_training_data_deduped_enriched.csv`
Expected: prints held-out counts, `new Brier`/`old Brier` (equal here since same file), a replay line, and a `GATE:` verdict. (With `--old == --new`, `brier_ok` is False — that's fine; this step only verifies the script executes end-to-end.)

- [ ] **Step 4: Commit**

```bash
git add validate_retrain.py
git commit -m "Add validate_retrain.py: out-of-sample gate for retrained model

Held-out (most-recent 20% of markets) Brier-beats-old AND current-rule
(THRESH=0.25, p_raw<0.20 NO gate) EV>0, using the new model's own
re-predicted probabilities. Exit 0=PASS gates the deploy + size re-enable."
```

---

## Operational Runbook (Phase 1 & 2 — you run these; not code tasks)

These steps require production access and the exported data; they are performed after Tasks 1–5 are merged and deployed.

**Deploy Phase 0 (after merge):**
1. Deploy the branch to Render.
2. On the Render shell: `python3 apply_backtest_settings.py`
3. Verify: `python3 diag.py` → confirm `mispricing_threshold = 0.2500`, `no_max_p_raw = 0.20`, `live_max_contracts = 1`, `signal_mode = ensemble`. Restart the scheduler.
4. Confirm the seed no longer reverts: restart once more and re-run `diag.py` — `mispricing_threshold` must remain `0.25`.

**Phase 1 (retrain + validate):**
5. Export fresh data: `curl -s "<app-url>/api/export/live-training-data" -o live_training_data.csv` (or Analytics → Export in the UI).
6. Back up the current model: `cp raw_feature_model.pkl raw_feature_model.prev.pkl`
7. Retrain: `python3 merge_and_retrain.py --live-data live_training_data.csv`
8. Validate: `python3 validate_retrain.py --new raw_feature_model.pkl --old raw_feature_model.prev.pkl --data live_training_data.csv`
9. If `GATE: FAIL` → iterate (try `--live-only`, adjust `--live-weight`, review drift output). Do NOT deploy a failing model.
10. If `GATE: PASS` → upload: `POST <app-url>/api/model/upload` with the new `.pkl`; then `POST <app-url>/api/model/reload`.

**Phase 2 (restore full sizing):**
11. Once the passing model is live and stable, clear the interim cap: set `live_max_contracts` to empty (via settings UI/API), leaving the `$4` cap and `max_daily_loss` as backstops.
12. Monitor `/api/live/stats` and the live trade log.

---

## Self-Review

**Spec coverage:**
- Cause 1 (oversizing) → Task 3 (`live_max_contracts`) + Phase 2 gating. ✓
- Cause 2 (threshold drift) → Task 1 (seed guard fix) + Task 4 (deploy) + runbook verify. ✓
- Cause 3 (NO miscalibration) → Task 2 (`no_max_p_raw` gate). ✓
- Cause 4 (edge decay) → Task 5 (`validate_retrain.py`) + runbook Phase 1. ✓
- Interim tiny-fixed-size posture → Task 3 + Task 4 (`live_max_contracts=1`). ✓
- Validation gate (Brier + EV) → Task 5. ✓

**Placeholder scan:** No TBD/TODO; every code step shows full code. ✓

**Type consistency:** `no_max_p_raw: float` param name identical across `evaluate_ensemble_signal`, `evaluate_mispricing_signal`, and `evaluate_live_signal` read. `_apply_contract_cap(contracts, live_max_contracts) -> int` name identical in helper, call site, and tests. `validate_retrain.py` reuses `entry_cost`/`decide` signatures mirrored from `backtest_current_rule.py`. ✓

**Note carried into execution:** if any existing `test_signal_engine.py` case triggers a bearish NO with `p_raw >= 0.20`, it will now return NO SIGNAL; update that case to pass `no_max_p_raw=1.0` (documented in Task 2 Step 6).
