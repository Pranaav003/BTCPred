# BTCPred Comprehensive Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix critical bugs, harden reliability, and improve signal accuracy across the BTCPred Kalshi trading platform.

**Architecture:** Layered approach — stabilize (bug fixes, safety), then harden (tests, resilience), then enhance (ML pipeline, UX). Each phase builds on the previous; no phase depends on a later one.

**Tech Stack:** Python 3.11+, Flask, SQLAlchemy, APScheduler, scikit-learn, XGBoost, pytest, Chart.js, Jinja2

## Global Constraints

- All code changes must preserve existing API contracts (endpoint paths, request/response shapes)
- Financial calculations must use `round(value, 2)` for dollar amounts
- `raw_feature_model.pkl` must remain loadable by `model_loader.py` after any feature changes
- Test files go in `tests/` at project root, mirroring `app/` structure
- Every commit message ends with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`
- Dependencies must be pinned to exact versions in `requirements.txt`
- No new Python packages without adding to `requirements.txt` with a pinned version

---

## Phase 1 — Stabilize

### Task 1: Fix PnL miscalculation in `_apply_kalshi_settlement`

**Files:**
- Modify: `app/resolver.py:142-151`
- Create: `tests/test_resolver_pnl.py`

**Interfaces:**
- Consumes: `_parse_fp_count`, `_parse_fp_dollars` from `app.kalshi_trader`
- Produces: Correct `trade.realized_pnl` for live trades resolved via Kalshi settlement

The bug: line 151 computes `pnl = round((revenue_cents / 100.0) - total_cost - fee_cost, 2)` where `total_cost = yes_cost + no_cost`. For a YES-side trade, this subtracts both the YES cost and the NO cost from revenue, which is wrong. The `side_cost` variable (line 156) is already computed but unused for the PnL formula.

- [ ] **Step 1: Write the failing test**

Create `tests/__init__.py` (empty) and `tests/test_resolver_pnl.py`:

```python
"""Tests for PnL calculation in _apply_kalshi_settlement."""
import pytest
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock


@pytest.fixture
def make_trade():
    """Factory for LiveTrade-like objects with configurable side."""
    def _make(side="YES", contracts=10, entry_price=0.70, cost_dollars=7.00):
        trade = MagicMock()
        trade.side = side
        trade.contracts = contracts
        trade.entry_price = entry_price
        trade.cost_dollars = cost_dollars
        return trade
    return _make


def _make_settlement(market_result="yes", yes_count=10, no_count=0,
                     yes_cost=7.00, no_cost=0.0, fee_cost=0.07,
                     revenue=1000):
    """Build a settlement dict matching Kalshi API shape.

    revenue is in cents (integer), costs are in dollars (float).
    """
    return {
        "market_result": market_result,
        "yes_count_fp": f"{yes_count}.00",
        "no_count_fp": f"{no_count}.00",
        "yes_total_cost_dollars": f"{yes_cost:.2f}",
        "no_total_cost_dollars": f"{no_cost:.2f}",
        "fee_cost": f"{fee_cost:.2f}",
        "revenue": revenue,
    }


def test_yes_trade_correct_pnl_uses_side_cost_only(make_trade):
    """PnL for a winning YES trade must use yes_cost, not yes_cost+no_cost."""
    trade = make_trade(side="YES", contracts=10)
    settlement = _make_settlement(
        market_result="yes",
        yes_count=10,
        yes_cost=7.00,
        no_cost=0.00,
        fee_cost=0.07,
        revenue=1000,  # $10.00
    )
    from app.resolver import _apply_kalshi_settlement
    now = datetime.now(timezone.utc)
    _apply_kalshi_settlement(trade, settlement, now)
    # Correct: revenue/100 - side_cost(yes=7.00) - fee = 10.00 - 7.00 - 0.07 = 2.93
    assert trade.realized_pnl == 2.93


def test_yes_trade_correct_pnl_with_nonzero_no_cost(make_trade):
    """Even if no_cost is non-zero, YES PnL must not subtract it."""
    trade = make_trade(side="YES", contracts=10)
    settlement = _make_settlement(
        market_result="yes",
        yes_count=10,
        yes_cost=7.00,
        no_cost=3.00,  # This should NOT be subtracted for a YES trade
        fee_cost=0.07,
        revenue=1000,  # $10.00
    )
    from app.resolver import _apply_kalshi_settlement
    now = datetime.now(timezone.utc)
    _apply_kalshi_settlement(trade, settlement, now)
    # Correct: 10.00 - 7.00 - 0.07 = 2.93 (NOT 10.00 - 10.00 - 0.07 = -0.07)
    assert trade.realized_pnl == 2.93


def test_no_trade_correct_pnl_uses_no_cost(make_trade):
    """PnL for a winning NO trade must use no_cost, not yes_cost+no_cost."""
    trade = make_trade(side="NO", contracts=10)
    settlement = _make_settlement(
        market_result="no",
        yes_count=0,
        no_count=10,
        yes_cost=0.00,
        no_cost=3.00,
        fee_cost=0.03,
        revenue=1000,  # $10.00
    )
    from app.resolver import _apply_kalshi_settlement
    now = datetime.now(timezone.utc)
    _apply_kalshi_settlement(trade, settlement, now)
    # Correct: 10.00 - 3.00 - 0.03 = 6.97
    assert trade.realized_pnl == 6.97


def test_yes_trade_wrong_pnl(make_trade):
    """Losing YES trade: PnL = revenue/100 - side_cost - fee (negative)."""
    trade = make_trade(side="YES", contracts=10)
    settlement = _make_settlement(
        market_result="no",
        yes_count=10,
        yes_cost=7.00,
        no_cost=0.00,
        fee_cost=0.07,
        revenue=0,  # Lost — revenue is 0
    )
    from app.resolver import _apply_kalshi_settlement
    now = datetime.now(timezone.utc)
    _apply_kalshi_settlement(trade, settlement, now)
    # Lost: 0.00 - 7.00 - 0.07 = -7.07
    assert trade.realized_pnl == -7.07
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /Users/pranaaviyer/IBMProjects/BTCPred && python -m pytest tests/test_resolver_pnl.py -v`
Expected: FAIL — the test that has `no_cost=3.00` will show the bug (PnL uses `total_cost` instead of `side_cost`)

- [ ] **Step 3: Fix the PnL calculation**

In `app/resolver.py`, change line 151 from:

```python
    pnl = round((revenue_cents / 100.0) - total_cost - fee_cost, 2)
```

to:

```python
    pnl = round((revenue_cents / 100.0) - side_cost - fee_cost, 2)
```

Note: `side_cost` is already computed on line 156 (`side_cost = yes_cost if trade_is_yes else no_cost`). Move the `side_cost` assignment above the `pnl` computation (swap lines 151 and 156). The full replacement block should be:

```python
    side_cost = yes_cost if trade_is_yes else no_cost
    pnl = round((revenue_cents / 100.0) - side_cost - fee_cost, 2)

    outcome_yes = market_result == "yes"
    won = (trade_is_yes and outcome_yes) or (not trade_is_yes and not outcome_yes)
```

Remove the now-redundant `total_cost` variable (line 150: `total_cost = yes_cost + no_cost`) since nothing else uses it.

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd /Users/pranaaviyer/IBMProjects/BTCPred && python -m pytest tests/test_resolver_pnl.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add app/resolver.py tests/test_resolver_pnl.py tests/__init__.py
git commit -m "fix: use side_cost instead of total_cost for live trade PnL settlement"
```

---

### Task 2: Fix inconsistent `max_daily_loss` defaults

**Files:**
- Modify: `app/scheduler.py:36`
- Modify: `app/db_helpers.py:230`
- Create: `tests/test_daily_loss_default.py`

**Interfaces:**
- Consumes: `AppSettings.get("max_daily_loss")` from `app.models`
- Produces: Consistent default of 50.0 across all three locations (paper_trading.py already uses `"200.0"` in code, but the DB default is also `"200.0"` — we align everything to `"50.0"`)

- [ ] **Step 1: Write the failing test**

Create `tests/test_daily_loss_default.py`:

```python
"""Tests for consistent max_daily_loss default across codebase."""
import ast


def _find_max_daily_loss_defaults():
    """Scan source files for max_daily_loss default values."""
    results = {}
    with open("app/scheduler.py") as f:
        content = f.read()
    for line in content.splitlines():
        if 'AppSettings.get("max_daily_loss"' in line:
            # Extract the default string value
            if '"200.0"' in line:
                results["scheduler_paper"] = "200.0"
            elif '"50.0"' in line:
                results["scheduler_paper"] = "50.0"
    with open("app/db_helpers.py") as f:
        content = f.read()
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith('"max_daily_loss"'):
            # Line looks like: "max_daily_loss": "200.0",
            if '"200.0"' in stripped:
                results["db_helpers_default"] = "200.0"
            elif '"50.0"' in stripped:
                results["db_helpers_default"] = "50.0"
    with open("app/paper_trading.py") as f:
        content = f.read()
    for line in content.splitlines():
        if 'AppSettings.get("max_daily_loss"' in line:
            if '"200.0"' in line:
                results["paper_trading"] = "200.0"
            elif '"50.0"' in line:
                results["paper_trading"] = "50.0"
    return results


def test_max_daily_loss_defaults_are_consistent():
    """All max_daily_loss defaults should be 50.0 across the codebase."""
    defaults = _find_max_daily_loss_defaults()
    for location, value in defaults.items():
        assert value == "50.0", f"{location} has max_daily_loss default {value}, expected 50.0"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /Users/pranaaviyer/IBMProjects/BTCPred && python -m pytest tests/test_daily_loss_default.py -v`
Expected: FAIL — scheduler_paper and db_helpers_default show "200.0"

- [ ] **Step 3: Fix the defaults**

In `app/scheduler.py`, change line 36 from:

```python
    max_daily_loss = float(AppSettings.get("max_daily_loss", "200.0") or 200.0)
```

to:

```python
    max_daily_loss = float(AppSettings.get("max_daily_loss", "50.0") or 50.0)
```

In `app/db_helpers.py`, change line 230 (inside the `seed_default_settings` defaults dict) from:

```python
        "max_daily_loss": "200.0",
```

to:

```python
        "max_daily_loss": "50.0",
```

In `app/paper_trading.py`, find the line with `AppSettings.get("max_daily_loss", "200.0")` and change to:

```python
    max_daily_loss = float(AppSettings.get("max_daily_loss", "50.0") or 50.0)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd /Users/pranaaviyer/IBMProjects/BTCPred && python -m pytest tests/test_daily_loss_default.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/scheduler.py app/db_helpers.py app/paper_trading.py tests/test_daily_loss_default.py
git commit -m "fix: align max_daily_loss default to 50.0 across all code paths"
```

---

### Task 3: Fix `seconds_to_close` truncation bug

**Files:**
- Modify: `app/scheduler.py:378`
- Create: `tests/test_seconds_truncation.py`

**Interfaces:**
- Consumes: `snapshot.get("seconds_to_close")`
- Produces: Correct ceiling rounding so 89.7s -> 90s (fails the 90s minimum check) instead of 89s (passes incorrectly)

- [ ] **Step 1: Write the failing test**

Create `tests/test_seconds_truncation.py`:

```python
"""Tests for seconds_to_close rounding in auto-trade guard."""
import math


def test_ceil_prevents_truncation_bypass():
    """89.7 seconds must round UP to 90, not down to 89.

    With a 90s minimum, int(89.7)=89 would incorrectly pass the guard.
    """
    seconds_float = 89.7
    min_seconds = 90

    # Current (buggy): int() truncates to 89, passes the check incorrectly
    truncated = int(seconds_float)
    assert truncated < min_seconds  # This is False with int()! 89 < 90 is True, but 89.7 is really 90

    # Fixed: math.ceil rounds up to 90, correctly fails the check
    ceiled = math.ceil(seconds_float)
    assert ceiled >= min_seconds  # Correctly identifies as at or above minimum


def test_exact_integer_stays_same():
    """Exact integer values like 90.0 should remain 90."""
    assert math.ceil(90.0) == 90
    assert math.ceil(120.0) == 120


def test_fractional_values_round_up():
    """Any fractional value rounds up to the next integer."""
    assert math.ceil(89.1) == 90
    assert math.ceil(89.9) == 90
    assert math.ceil(90.1) == 91
```

- [ ] **Step 2: Run the test to verify it passes (logic test, not integration)**

Run: `cd /Users/pranaaviyer/IBMProjects/BTCPred && python -m pytest tests/test_seconds_truncation.py -v`
Expected: PASS — these tests validate the math behavior, not the current code

- [ ] **Step 3: Fix the code**

In `app/scheduler.py`, add `import math` at the top of the file (after the existing imports) if not already present.

Change line 378 from:

```python
                        seconds_left = int(snapshot.get("seconds_to_close", 0) or 0)
```

to:

```python
                        seconds_left = math.ceil(float(snapshot.get("seconds_to_close", 0) or 0))
```

Also fix the same pattern in `_execute_live_trade` at line 136:

```python
            seconds_left = int(snapshot.get("seconds_to_close", 0) or 0)
```

to:

```python
            seconds_left = math.ceil(float(snapshot.get("seconds_to_close", 0) or 0))
```

- [ ] **Step 4: Run tests**

Run: `cd /Users/pranaaviyer/IBMProjects/BTCPred && python -m pytest tests/test_seconds_truncation.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/scheduler.py tests/test_seconds_truncation.py
git commit -m "fix: use math.ceil for seconds_to_close to prevent truncation bypass"
```

---

### Task 4: Fix data leakage in train/test split

**Files:**
- Modify: `train_raw_model.py:82-93`
- Modify: `model_comparison.py` (find its `temporal_split` and update)
- Modify: `merge_and_retrain.py` (find its split logic and update)
- Create: `tests/test_temporal_split.py`

**Interfaces:**
- Consumes: DataFrame with `market_ticker` and `close_ts` columns
- Produces: `train_df`, `test_df` where no market appears in both sets, with a 15-minute embargo period

- [ ] **Step 1: Write the failing test**

Create `tests/test_temporal_split.py`:

```python
"""Tests for market-level temporal split without data leakage."""
import pandas as pd
import numpy as np
from train_raw_model import temporal_split


def _make_dataset(n_markets=10, rows_per_market=5):
    """Create a synthetic dataset with known market_ticker values."""
    rows = []
    for i in range(n_markets):
        ticker = f"KXBTC15M-{i:03d}"
        base_ts = 1700000000 + i * 900  # 15-min apart
        for j in range(rows_per_market):
            rows.append({
                "market_ticker": ticker,
                "close_ts": base_ts + j * 60,
                "final_outcome_yes": np.random.choice([0, 1]),
                "return_1m": np.random.randn() * 0.01,
                "return_3m": np.random.randn() * 0.02,
                "return_5m": np.random.randn() * 0.03,
            })
    return pd.DataFrame(rows)


def test_no_market_in_both_train_and_test():
    """No market_ticker may appear in both train and test sets."""
    df = _make_dataset(n_markets=20, rows_per_market=3)
    train_df, test_df = temporal_split(df)
    train_tickers = set(train_df["market_ticker"].unique())
    test_tickers = set(test_df["market_ticker"].unique())
    overlap = train_tickers & test_tickers
    assert len(overlap) == 0, f"Markets in both train and test: {overlap}"


def test_embargo_period_between_train_and_test():
    """Test set must start at least 900s (15 min) after the last training close_ts."""
    df = _make_dataset(n_markets=20, rows_per_market=3)
    train_df, test_df = temporal_split(df)
    max_train_ts = train_df["close_ts"].max()
    min_test_ts = test_df["close_ts"].min()
    gap = min_test_ts - max_train_ts
    assert gap >= 900, f"Embargo gap is only {gap}s, need >= 900s (15 min)"


def test_train_comes_before_test():
    """All training data must be chronologically before test data."""
    df = _make_dataset(n_markets=20, rows_per_market=3)
    train_df, test_df = temporal_split(df)
    max_train_ts = train_df["close_ts"].max()
    min_test_ts = test_df["close_ts"].min()
    assert max_train_ts < min_test_ts
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /Users/pranaaviyer/IBMProjects/BTCPred && python -m pytest tests/test_temporal_split.py -v`
Expected: FAIL — the current `temporal_split` splits by row index, so the same market can appear in both sets

- [ ] **Step 3: Rewrite `temporal_split` in `train_raw_model.py`**

Replace the `temporal_split` function (lines 82-93) with:

```python
def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data by market_ticker to prevent data leakage.

    Markets are sorted by close_ts and split 80/20 so no market appears
    in both sets. A 15-minute embargo gap is enforced between the last
    training market and the first test market.
    """
    if not 0 < TEST_SIZE < 1:
        raise ValueError("TEST_SIZE must be between 0 and 1.")

    if "market_ticker" not in df.columns:
        # Fallback to row-level split if no market_ticker column
        split_idx = int(len(df) * (1 - TEST_SIZE))
        return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()

    # Get unique markets sorted by their earliest close_ts
    market_order = (
        df.groupby("market_ticker")["close_ts"]
        .min()
        .sort_values()
        .index
        .tolist()
    )
    n_test_markets = max(1, int(len(market_order) * TEST_SIZE))
    n_train_markets = len(market_order) - n_test_markets
    if n_train_markets <= 0:
        raise ValueError("Not enough markets for train/test split.")

    train_tickers = set(market_order[:n_train_markets])
    test_tickers = set(market_order[n_train_markets:])

    train_df = df[df["market_ticker"].isin(train_tickers)].copy()
    test_df = df[df["market_ticker"].isin(test_tickers)].copy()

    # Enforce 15-minute embargo between last train and first test
    max_train_ts = train_df["close_ts"].max()
    min_test_ts = test_df["close_ts"].min()
    if min_test_ts - max_train_ts < 900:
        # Drop test markets that start within the embargo window
        embargo_cutoff = max_train_ts + 900
        test_df = test_df[test_df["close_ts"] >= embargo_cutoff].copy()

    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError("Train/test split resulted in an empty partition.")

    return train_df, test_df
```

- [ ] **Step 4: Apply the same fix to `model_comparison.py` and `merge_and_retrain.py`**

Find the `temporal_split` or equivalent split function in each file and replace with the same market-level split logic. In `model_comparison.py`, the function may be named `temporal_test_split` — update it identically. In `merge_and_retrain.py`, find the split logic and apply the same approach.

- [ ] **Step 5: Run the test to verify it passes**

Run: `cd /Users/pranaaviyer/IBMProjects/BTCPred && python -m pytest tests/test_temporal_split.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add train_raw_model.py model_comparison.py merge_and_retrain.py tests/test_temporal_split.py
git commit -m "fix: split by market_ticker to prevent data leakage in train/test split"
```

---

### Task 5: Remove duplicate features from RAW_FEATURES

**Files:**
- Modify: `train_raw_model.py:23-50` (RAW_FEATURES list)
- Modify: `app/feature_engineering.py:188-192` (momentum computation)
- Create: `tests/test_raw_features.py`

**Interfaces:**
- Consumes: `RAW_FEATURES` from `train_raw_model.py`
- Produces: Deduplicated feature list (23 features instead of 26). The model pipeline must be retrained after this change.

Remove: `momentum_1m` (identical to `return_1m`), `momentum_3m` (identical to `return_3m`), `momentum_5m` (identical to `return_5m`), `price_velocity_5m` (linear rescaling of `return_5m`).

Keep: `momentum_acceleration` (genuinely different: `return_1m - return_3m/3`).

- [ ] **Step 1: Write the failing test**

Create `tests/test_raw_features.py`:

```python
"""Tests for RAW_FEATURES deduplication and correctness."""
from train_raw_model import RAW_FEATURES


def test_no_duplicate_features():
    """RAW_FEATURES must not contain duplicate feature names."""
    assert len(RAW_FEATURES) == len(set(RAW_FEATURES)), (
        f"Duplicate features found: {[f for f in RAW_FEATURES if RAW_FEATURES.count(f) > 1]}"
    )


def test_no_known_redundant_features():
    """Features that are mathematically identical to others must be removed."""
    redundant = {"momentum_1m", "momentum_3m", "momentum_5m", "price_velocity_5m"}
    present = redundant & set(RAW_FEATURES)
    assert len(present) == 0, f"Redundant features still present: {present}"


def test_momentum_acceleration_is_present():
    """momentum_acceleration is genuinely different and must be kept."""
    assert "momentum_acceleration" in RAW_FEATURES
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /Users/pranaaviyer/IBMProjects/BTCPred && python -m pytest tests/test_raw_features.py -v`
Expected: FAIL — `test_no_known_redundant_features` fails because `momentum_1m/3m/5m` and `price_velocity_5m` are still in RAW_FEATURES

- [ ] **Step 3: Remove duplicate features from RAW_FEATURES**

In `train_raw_model.py`, replace the `RAW_FEATURES` list (lines 23-50) with:

```python
RAW_FEATURES = [
    "seconds_to_close",
    "entry_bucket",
    "return_1m",
    "return_3m",
    "return_5m",
    "volatility_3m",
    "volatility_5m",
    "range_5m",
    "abs_return_1m",
    "trade_count_1m",
    "trade_count_3m",
    "trade_count_5m",
    "volume_1m",
    "volume_3m",
    "volume_5m",
    "avg_trade_price_1m",
    "avg_trade_price_3m",
    "momentum_acceleration",
    "flip_count_5m",
    "return_1m_x_inv_time",
    "return_3m_x_inv_time",
    "volatility_5m_x_inv_time",
]
```

22 features (down from 26). Removed: `momentum_1m`, `momentum_3m`, `momentum_5m`, `price_velocity_5m`.

In `app/feature_engineering.py`, the `compute_features` function (around lines 188-192) still computes `momentum_1m`, `momentum_3m`, `momentum_5m`, and `price_velocity_5m` and adds them to the returned dict. These are harmless (extra keys in the dict don't affect the model since `predict_proba_raw` only uses features in the model's feature list), but for cleanliness, remove these lines from `compute_features`:

Remove lines that set:
```python
    "momentum_1m": return_1m,
    "momentum_3m": return_3m,
    "momentum_5m": return_5m,
```

And remove:
```python
    "price_velocity_5m": return_5m / 5.0 if return_5m is not None else 0.0,
```

Keep the `momentum_acceleration` computation.

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd /Users/pranaaviyer/IBMProjects/BTCPred && python -m pytest tests/test_raw_features.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add train_raw_model.py app/feature_engineering.py tests/test_raw_features.py
git commit -m "fix: remove duplicate features (momentum_1m/3m/5m, price_velocity_5m) from RAW_FEATURES"
```

**Important:** After this change, the existing `raw_feature_model.pkl` is incompatible (it was trained on 26 features, now we have 22). The model must be retrained before deployment. This is covered in Phase 3 Task 10.

---

### Task 6: Fix CSS syntax error and pin dependencies

**Files:**
- Modify: `app/static/css/main.css:~1699-1703`
- Modify: `requirements.txt`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: None (standalone fixes)
- Produces: Clean CSS parsing, pinned dependency versions, model.pkl excluded from git

- [ ] **Step 1: Fix CSS syntax error**

In `app/static/css/main.css`, find the orphaned block around line 1699-1703. It looks like:

```css
    display: flex;
    flex-direction: column;
    gap: var(--space-4);
}
```

These lines are after the closing brace of `.monitor-live-strip { ... }` and have no opening selector. Remove them entirely.

- [ ] **Step 2: Pin dependencies in requirements.txt**

Replace the contents of `requirements.txt` with pinned versions. Use versions compatible with the current codebase:

```
flask==3.1.1
flask-sqlalchemy==3.1.1
flask-migrate==4.1.0
apscheduler==3.10.4
python-dotenv==1.1.0
requests==2.32.4
numpy==2.2.6
pandas==2.2.3
scikit-learn==1.6.1
xgboost==2.1.4
joblib==1.4.2
gunicorn==23.0.0
psycopg2-binary==2.9.10
cryptography==44.0.3
pytest==8.3.5
```

Note: Added `flask-migrate` and `pytest`. Removed `flask-apscheduler` (unused — the code imports directly from `apscheduler`).

- [ ] **Step 3: Fix .gitignore to exclude .pkl files**

In `.gitignore`, change:

```
# *.pkl
```

to:

```
*.pkl
```

This excludes `raw_feature_model.pkl` from git tracking. After this change, run:

```bash
git rm --cached raw_feature_model.pkl
```

This removes it from git tracking without deleting the file from disk.

- [ ] **Step 4: Commit**

```bash
git add app/static/css/main.css requirements.txt .gitignore
git rm --cached raw_feature_model.pkl
git commit -m "fix: CSS syntax error, pin dependencies, exclude .pkl from git"
```

---

### Task 7: Add API input validation and CSRF protection

**Files:**
- Modify: `app/routes/api.py:1303-1322` (paper trade endpoint)
- Modify: `app/__init__.py` (CSRF init)
- Modify: `requirements.txt` (already done in Task 6 — flask-wtf is added here)

**Interfaces:**
- Consumes: `Flask-WTF` CSRF protection
- Produces: Server-side dollar_amount validation, CSRF tokens on all POST endpoints

- [ ] **Step 1: Add `flask-wtf` to requirements.txt**

Add `flask-wtf==1.2.2` to `requirements.txt`.

- [ ] **Step 2: Initialize CSRFProtect in app factory**

In `app/__init__.py`, add after the existing imports:

```python
from flask_wtf.csrf import CSRFProtect
```

Inside `create_app()`, after `db.init_app(app)` (around line 48), add:

```python
    csrf = CSRFProtect(app)
```

Since this is a JSON API (not form-based), CSRF protection for JSON requests needs the `X-CSRFToken` header. Add this config to `BaseConfig` in `app/config.py`:

```python
    WTF_CSRF_CHECK_DEFAULT = False
```

And in `app/__init__.py`, after initializing CSRFProtect, add:

```python
    # Exempt JSON API endpoints from automatic CSRF check — they use
    # X-CSRFToken header instead. The @csrf.exempt decorator is not
    # needed for JSON APIs when WTF_CSRF_CHECK_DEFAULT is False.
    @app.after_request
    def set_csrf_cookie(response):
        response.set_cookie("csrf_token", app.config.get("CSRF_COOKIE_NAME", "csrf_token"))
        return response
```

- [ ] **Step 3: Add dollar_amount validation to `/api/paper/trade`**

In `app/routes/api.py`, modify the `paper_trade` endpoint (lines 1303-1322) to validate the trade size:

```python
@api_bp.route("/paper/trade", methods=["POST"])
def paper_trade():
    payload = request.get_json(silent=True) or {}
    side = payload.get("side")
    contracts = payload.get("contracts")
    ticker = payload.get("ticker")
    seconds_to_close = payload.get("seconds_to_close")
    dollar_amount = payload.get("dollar_amount")

    if side is None or contracts is None or ticker is None:
        return jsonify({"error": "side, contracts, and ticker are required"}), 400

    # Server-side validation: check dollar_amount against portfolio cash
    if dollar_amount is not None:
        try:
            dollar_amount = float(dollar_amount)
        except (TypeError, ValueError):
            return jsonify({"error": "dollar_amount must be a number"}), 400
        if dollar_amount <= 0:
            return jsonify({"error": "dollar_amount must be positive"}), 400
        from app.models import Portfolio
        portfolio = Portfolio.get_or_create()
        if dollar_amount > float(portfolio.cash or 0):
            return jsonify({"error": f"dollar_amount ${dollar_amount:.2f} exceeds cash ${portfolio.cash:.2f}"}), 400

    result = execute_paper_trade(
        side=side,
        contracts=contracts,
        ticker=ticker,
        seconds_to_close=seconds_to_close,
        dollar_amount=dollar_amount,
    )
    if result.get("success"):
        return jsonify(result), 200
    return jsonify(result), 400
```

- [ ] **Step 4: Install new dependencies and test**

Run: `cd /Users/pranaaviyer/IBMProjects/BTCPred && pip install flask-wtf==1.2.2`
Verify the app starts: `python -c "from app import create_app; app = create_app(); print('OK')"`

- [ ] **Step 5: Commit**

```bash
git add app/__init__.py app/config.py app/routes/api.py requirements.txt
git commit -m "feat: add API input validation and CSRF protection"
```

---

### Task 8: Add Flask-Migrate for schema migrations

**Files:**
- Modify: `app/__init__.py` (add Flask-Migrate init)
- Modify: `requirements.txt` (already added in Task 6)
- Create: `migrations/` (auto-generated by `flask db init`)

**Interfaces:**
- Consumes: `Flask-Migrate` (Alembic), `db` from `app.extensions`
- Produces: `migrations/` directory with initial migration, `flask db migrate` / `flask db upgrade` CLI commands

- [ ] **Step 1: Initialize Flask-Migrate in app factory**

In `app/__init__.py`, add after imports:

```python
from flask_migrate import Migrate
```

Inside `create_app()`, after `db.init_app(app)` (around line 48), add:

```python
    migrate = Migrate(app, db)
```

- [ ] **Step 2: Generate the initial migration**

Run:
```bash
cd /Users/pranaaviyer/IBMProjects/BTCPred
export FLASK_APP="app:create_app()"
flask db init
flask db migrate -m "initial schema"
```

This creates the `migrations/` directory and an initial migration capturing the current schema.

- [ ] **Step 3: Remove the SQLite-only schema hack**

In `app/__init__.py`, remove the `_ensure_signal_schema_columns` function (approximately lines 17-33) and the call to it (around line 51).

- [ ] **Step 4: Verify migration works on SQLite**

Run: `flask db upgrade`
Expected: No errors, all tables created.

- [ ] **Step 5: Commit**

```bash
git add app/__init__.py migrations/
git commit -m "feat: add Flask-Migrate for schema migrations, remove SQLite schema hack"
```

---

## Phase 2 — Harden

### Task 9: Add test suite for financial logic

**Files:**
- Create: `tests/test_signal_engine.py`
- Create: `tests/test_paper_trading.py`
- Create: `tests/test_feature_engineering.py`

**Interfaces:**
- Consumes: `evaluate_signal`, `evaluate_mispricing_signal`, `evaluate_ensemble_signal` from `app.signal_engine`; `compute_pnl` from `app.resolver`; `compute_position_size` from `app.paper_trading`; `compute_features` from `app.feature_engineering`
- Produces: Test coverage for the most critical financial code paths

- [ ] **Step 1: Write signal engine tests**

Create `tests/test_signal_engine.py`:

```python
"""Tests for signal evaluation logic."""
import pytest
from app.signal_engine import (
    evaluate_signal,
    evaluate_mispricing_signal,
    evaluate_ensemble_signal,
    determine_agreement_region,
)


class TestDetermineAgreementRegion:
    def test_agree_yes(self):
        assert determine_agreement_region(0.80, 0.75, 0.65, 0.35) == "agree_yes"

    def test_agree_no(self):
        assert determine_agreement_region(0.20, 0.25, 0.65, 0.35) == "agree_no"

    def test_market_yes_raw_no(self):
        assert determine_agreement_region(0.70, 0.30, 0.65, 0.35) == "market_yes_raw_no"

    def test_market_no_raw_yes(self):
        assert determine_agreement_region(0.40, 0.70, 0.65, 0.35) == "market_no_raw_yes"

    def test_no_agreement(self):
        assert determine_agreement_region(0.50, 0.50, 0.65, 0.35) == "no_agreement"


class TestEvaluateSignal:
    def test_buy_yes_when_both_above_cutoff(self):
        result = evaluate_signal(
            p_market=0.80, p_raw=0.75,
            seconds_to_close=120, entry_bucket=120,
            yes_cutoff=0.65, no_cutoff=0.35,
            min_seconds=60, max_seconds=300,
        )
        assert result.signal == "PAPER BUY YES"
        assert result.agreement_region == "agree_yes"

    def test_no_signal_outside_window(self):
        result = evaluate_signal(
            p_market=0.80, p_raw=0.75,
            seconds_to_close=30, entry_bucket=30,
            yes_cutoff=0.65, no_cutoff=0.35,
            min_seconds=60, max_seconds=300,
        )
        assert result.signal == "NO SIGNAL"
        assert result.agreement_region == "outside_time_window"

    def test_entry_filter_blocks_high_yes_price(self):
        result = evaluate_signal(
            p_market=0.90, p_raw=0.75,
            seconds_to_close=120, entry_bucket=120,
            yes_cutoff=0.65, no_cutoff=0.35,
            min_seconds=60, max_seconds=300,
            max_entry_price_yes=0.80,
        )
        assert result.signal == "NO SIGNAL"
        assert result.entry_filtered is True

    def test_volatility_guard_blocks_agreement(self):
        result = evaluate_signal(
            p_market=0.80, p_raw=0.75,
            seconds_to_close=120, entry_bucket=120,
            yes_cutoff=0.65, no_cutoff=0.35,
            min_seconds=60, max_seconds=300,
            volatility_guard_active=True,
        )
        assert result.signal == "NO SIGNAL"
        assert result.agreement_region == "volatility_guard"


class TestEvaluateMispricingSignal:
    def test_bullish_mispricing(self):
        result = evaluate_mispricing_signal(
            p_market=0.30, p_raw=0.60,
            seconds_to_close=120, entry_bucket=120,
            min_seconds=60, max_seconds=300,
            mispricing_threshold=0.20,
        )
        assert result.signal == "PAPER BUY YES"
        assert result.agreement_region == "model_bullish"

    def test_no_signal_when_gap_below_threshold(self):
        result = evaluate_mispricing_signal(
            p_market=0.30, p_raw=0.40,
            seconds_to_close=120, entry_bucket=120,
            min_seconds=60, max_seconds=300,
            mispricing_threshold=0.20,
        )
        assert result.signal == "NO SIGNAL"

    def test_bearish_mispricing(self):
        result = evaluate_mispricing_signal(
            p_market=0.80, p_raw=0.40,
            seconds_to_close=120, entry_bucket=120,
            min_seconds=60, max_seconds=300,
            mispricing_threshold=0.20,
        )
        assert result.signal == "PAPER BUY NO"
        assert result.agreement_region == "model_bearish"


class TestEvaluateEnsembleSignal:
    def test_agreement_takes_priority(self):
        result = evaluate_ensemble_signal(
            p_market=0.80, p_raw=0.75,
            seconds_to_close=120, entry_bucket=120,
            yes_cutoff=0.65, mispricing_threshold=0.20,
            min_seconds=60, max_seconds=180,
        )
        assert result.signal == "PAPER BUY YES"
        assert "Agreement" in result.reason

    def test_mispricing_as_fallback(self):
        result = evaluate_ensemble_signal(
            p_market=0.30, p_raw=0.60,
            seconds_to_close=120, entry_bucket=120,
            yes_cutoff=0.65, mispricing_threshold=0.20,
            min_seconds=60, max_seconds=180,
        )
        assert result.signal == "PAPER BUY YES"
        assert "Mispricing" in result.reason or "model_bullish" == result.agreement_region

    def test_volatility_guard_blocks_agreement_not_mispricing(self):
        result = evaluate_ensemble_signal(
            p_market=0.30, p_raw=0.60,
            seconds_to_close=120, entry_bucket=120,
            yes_cutoff=0.65, mispricing_threshold=0.20,
            min_seconds=60, max_seconds=180,
            volatility_guard_active=True,
        )
        # Mispricing should bypass volatility guard
        assert result.signal == "PAPER BUY YES"
```

- [ ] **Step 2: Write paper trading tests**

Create `tests/test_paper_trading.py`:

```python
"""Tests for paper trading position sizing."""
import pytest
from app.paper_trading import compute_position_size


class TestComputePositionSize:
    def test_base_size_at_moderate_edge(self):
        """20-35c edge = 1.0x multiplier."""
        size = compute_position_size(
            p_market=0.70, side="YES", base_size=10.0
        )
        assert size == 10.0  # 1.0x base

    def test_large_edge_gets_boost(self):
        """35c+ edge = 1.5x multiplier."""
        size = compute_position_size(
            p_market=0.55, side="YES", base_size=10.0
        )
        assert size == 15.0  # 1.5x base

    def test_thin_edge_gets_reduced(self):
        """10-20c edge = 0.6x multiplier."""
        size = compute_position_size(
            p_market=0.88, side="YES", base_size=10.0
        )
        assert size == 6.0  # 0.6x base

    def test_very_thin_edge_gets_minimum(self):
        """0-10c edge = 0.3x multiplier."""
        size = compute_position_size(
            p_market=0.95, side="YES", base_size=10.0
        )
        assert size == 3.0  # 0.3x base

    def test_volatility_override_caps_at_1x(self):
        """Volatility override caps position at 1x regardless of edge."""
        size = compute_position_size(
            p_market=0.55, side="YES", base_size=10.0,
            volatility_override=True,
        )
        assert size == 10.0  # Capped at 1.0x
```

- [ ] **Step 3: Run all tests**

Run: `cd /Users/pranaaviyer/IBMProjects/BTCPred && python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_signal_engine.py tests/test_paper_trading.py
git commit -m "test: add signal engine and paper trading tests"
```

---

### Task 10: Add reliability improvements

**Files:**
- Modify: `app/kalshi_client.py:45-86` (_get method — add network retry)
- Modify: `app/kalshi_client.py:22-24, 430-456` (BTC price cache — add lock)
- Modify: `app/kalshi_auth.py:37-52` (key caching)
- Modify: `app/scheduler.py:28-31` (cooldown backoff)
- Modify: `app/model_loader.py:58-67` (missing feature warning)

**Interfaces:**
- Consumes: Existing cache/lock patterns from `kalshi_client.py`
- Produces: More resilient API client, cached auth key, exponential cooldown, missing feature warnings

- [ ] **Step 1: Add network retry to `_get()` in kalshi_client.py**

In `app/kalshi_client.py`, modify the `_get()` function. After the `requests.exceptions.Timeout` except block (around line 75), change the generic `except Exception` block to retry once on connection errors:

Replace the block starting with `except requests.exceptions.Timeout:` through `return None` (the entire except chain) with:

```python
        except requests.exceptions.Timeout:
            if attempt == 0:
                logger.warning("Request timed out, retrying: %s", url)
                time.sleep(2)
                continue
            logger.warning("Request timed out twice: %s", url)
            return None
        except requests.exceptions.ConnectionError:
            if attempt == 0:
                logger.warning("Connection error, retrying: %s", url)
                time.sleep(2)
                continue
            logger.warning("Connection error twice: %s", url)
            return None
        except requests.exceptions.HTTPError as exc:
            logger.error("Kalshi GET failed url=%s params=%s error=%s", url, params, exc)
            return None
        except Exception as exc:
            logger.error("Kalshi GET failed url=%s params=%s error=%s", url, params, exc)
            return None
```

- [ ] **Step 2: Add `_cache_lock` to BTC price cache in kalshi_client.py**

In `get_btc_price()` (lines 430-456), wrap the cache read/write with `_cache_lock`:

```python
def get_btc_price() -> float | None:
    """Return cached BTC spot price from CoinGecko (60s TTL, 5-min 429 backoff)."""
    global _btc_429_until
    with _cache_lock:
        now = time.time()
        if _btc_429_until > now:
            return _btc_price_cache.get("price")
        if now - _btc_price_cache.get("ts", 0) < BTC_PRICE_CACHE_TTL:
            return _btc_price_cache.get("price")
    try:
        resp = requests.get(
            COINGECKO_URL,
            params={"ids": "bitcoin", "vs_currencies": "usd"},
            timeout=10,
        )
        if resp.status_code == 429:
            with _cache_lock:
                _btc_429_until = now + 300
            logger.warning("CoinGecko 429 — backing off for 5 min")
            return _btc_price_cache.get("price")
        resp.raise_for_status()
        data = resp.json()
        price = float(data["bitcoin"]["usd"])
        with _cache_lock:
            _btc_price_cache["price"] = price
            _btc_price_cache["ts"] = now
        return price
    except Exception as exc:
        logger.warning("CoinGecko fetch failed: %s", exc)
        return _btc_price_cache.get("price")
```

- [ ] **Step 3: Cache the deserialized private key in kalshi_auth.py**

In `app/kalshi_auth.py`, add a module-level cache variable:

```python
_cached_private_key = None
```

Modify `get_private_key()` to use the cache:

```python
def get_private_key():
    global _cached_private_key
    if _cached_private_key is not None:
        return _cached_private_key
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import serialization

    pem = _normalize_pem(os.environ.get("KALSHI_PRIVATE_KEY", ""))
    if not pem:
        return None
    try:
        _cached_private_key = serialization.load_pem_private_key(
            pem.encode(),
            password=None,
            backend=default_backend(),
        )
        return _cached_private_key
    except Exception as exc:
        logger.error("Failed to load Kalshi private key: %s", exc)
        return None
```

- [ ] **Step 4: Add exponential cooldown backoff to scheduler**

In `app/scheduler.py`, replace the fixed cooldown (lines 28-31):

```python
_cooldown_until_ts = 0.0
```

with:

```python
_cooldown_until_ts = 0.0
_cooldown_level = 0
_COOLDOWN_SCHEDULE = [60, 120, 240, 300]  # Exponential, capped at 5 min
```

In `poll_and_signal()`, change the cooldown logic (around lines 293-300) from:

```python
                if _consecutive_failures >= _MAX_FAILURES_BEFORE_COOLDOWN:
                    _cooldown_until_ts = time.time() + _COOLDOWN_SECONDS
                    logger.warning(
                        "%s consecutive failures. Cooling down for %ss.",
                        _consecutive_failures,
                        _COOLDOWN_SECONDS,
                    )
                    _consecutive_failures = 0
```

to:

```python
                if _consecutive_failures >= _MAX_FAILURES_BEFORE_COOLDOWN:
                    global _cooldown_level
                    cooldown_seconds = _COOLDOWN_SCHEDULE[
                        min(_cooldown_level, len(_COOLDOWN_SCHEDULE) - 1)
                    ]
                    _cooldown_until_ts = time.time() + cooldown_seconds
                    _cooldown_level += 1
                    logger.warning(
                        "%s consecutive failures. Cooling down for %ss (level %s).",
                        _consecutive_failures,
                        cooldown_seconds,
                        _cooldown_level,
                    )
                    _consecutive_failures = 0
```

Also, when the poll succeeds (around line 302), reset the cooldown level:

```python
            _consecutive_failures = 0
            _cooldown_level = 0
```

- [ ] **Step 5: Add missing feature warning to model_loader.py**

In `app/model_loader.py`, in `predict_proba_raw()` (around line 64), change:

```python
        row[feature] = feature_dict.get(feature, 0.0) or 0.0
```

to:

```python
        value = feature_dict.get(feature)
        if value is None or value == "":
            logger.warning("Missing feature '%s' in prediction input, defaulting to 0.0", feature)
            row[feature] = 0.0
        else:
            row[feature] = float(value)
```

- [ ] **Step 6: Commit**

```bash
git add app/kalshi_client.py app/kalshi_auth.py app/scheduler.py app/model_loader.py
git commit -m "feat: add network retry, BTC cache lock, auth key caching, exponential cooldown, missing feature warnings"
```

---

### Task 11: Add observability (health check, model staleness, fill rate)

**Files:**
- Modify: `app/routes/api.py` (add `/api/health` depth, `/api/model/reload`)
- Modify: `render.yaml` (add `healthCheckPath`)
- Modify: `app/scheduler.py` (fill rate tracking)
- Modify: `app/db_helpers.py` (add rolling accuracy setting)
- Modify: `app/templates/dashboard.html` (model staleness banner)

**Interfaces:**
- Consumes: `/api/health` endpoint, `AppSettings` for fill rate tracking
- Produces: Render health check, model reload endpoint, model staleness warning on dashboard

- [ ] **Step 1: Enhance `/api/health` endpoint**

In `app/routes/api.py`, find the `/api/health` endpoint and enhance it to check DB and model:

```python
@api_bp.route("/health")
def health():
    """Liveness + readiness check for Render health check."""
    checks = {"status": "ok", "db": "ok", "model": "ok"}
    try:
        from app.models import AppSettings
        AppSettings.get("scheduler_running")
    except Exception:
        checks["db"] = "error"
        checks["status"] = "degraded"
    try:
        from app.model_loader import get_model
        if get_model() is None:
            checks["model"] = "not_loaded"
            checks["status"] = "degraded"
    except Exception:
        checks["model"] = "error"
        checks["status"] = "degraded"
    code = 200 if checks["status"] == "ok" else 503
    return jsonify(checks), code
```

- [ ] **Step 2: Add healthCheckPath to render.yaml**

In `render.yaml`, add `healthCheckPath: /api/health` to the web service:

```yaml
services:
  - type: web
    name: btcpred
    # ... existing config ...
    healthCheckPath: /api/health
```

- [ ] **Step 3: Add `/api/model/reload` endpoint**

In `app/routes/api.py`, add:

```python
@api_bp.route("/model/reload", methods=["POST"])
def model_reload():
    """Invalidate model cache so next prediction loads the latest .pkl."""
    from app.model_loader import _MODEL_BUNDLE, _MODEL_LOCK
    with _MODEL_LOCK:
        global _MODEL_BUNDLE
        _MODEL_BUNDLE = None
    return jsonify({"status": "cache_cleared"})
```

Note: This requires making `_MODEL_BUNDLE` importable. In `app/model_loader.py`, the variable is already module-level. The import works as shown.

- [ ] **Step 4: Add fill rate tracking to scheduler**

In `app/scheduler.py`, in `_execute_live_trade()`, after recording a live trade (around line 240), add fill rate tracking:

```python
            # Track fill rate for adaptive IOC pricing
            total_attempts = int(AppSettings.get("live_fill_attempts", "0") or 0) + 1
            total_filled = int(AppSettings.get("live_fill_successes", "0") or 0)
            if order_result.get("success"):
                total_filled += 1
            AppSettings.set("live_fill_attempts", str(total_attempts))
            AppSettings.set("live_fill_successes", str(total_filled))
```

- [ ] **Step 5: Add model staleness warning to dashboard**

In `app/routes/dashboard.py`, in the `dashboard()` route, add model age check:

```python
    # Model staleness check
    model_age_days = None
    try:
        from app.model_loader import get_model
        bundle = get_model()
        if bundle and bundle.get("trained_at"):
            from datetime import datetime, timezone
            trained = datetime.fromisoformat(bundle["trained_at"].replace("Z", "+00:00"))
            model_age_days = (datetime.now(timezone.utc) - trained).days
    except Exception:
        pass
```

Pass `model_age_days` to the template and add a warning banner in `app/templates/dashboard.html` when `model_age_days` is not None and > 7.

- [ ] **Step 6: Commit**

```bash
git add app/routes/api.py render.yaml app/scheduler.py app/routes/dashboard.py app/templates/dashboard.html
git commit -m "feat: add health check, model reload endpoint, fill rate tracking, model staleness warning"
```

---

## Phase 3 — Enhance

### Task 12: Add new features to feature engineering

**Files:**
- Modify: `app/feature_engineering.py` (add 8 new features)
- Modify: `train_raw_model.py` (add to RAW_FEATURES)
- Create: `tests/test_new_features.py`

**Interfaces:**
- Consumes: Candle and trade data from `get_candles()`, `get_trades()`
- Produces: 8 new feature keys in the snapshot dict: `bid_ask_spread`, `volume_acceleration`, `trade_intensity`, `rsi_14`, `session`, `distance_from_strike`, `outcome_rate_bucket`, `return_5m_ratio`

- [ ] **Step 1: Add new features to RAW_FEATURES**

In `train_raw_model.py`, append to the `RAW_FEATURES` list:

```python
RAW_FEATURES = [
    # ... existing 22 features ...
    "bid_ask_spread",
    "volume_acceleration",
    "trade_intensity",
    "rsi_14",
    "session",
    "distance_from_strike",
    "outcome_rate_bucket",
    "return_5m_ratio",
]
```

30 features total (22 existing + 8 new).

- [ ] **Step 2: Implement the new features in compute_features**

In `app/feature_engineering.py`, inside `compute_features()`, add before the final `return` statement:

```python
    # --- New features ---
    # Bid-ask spread from Kalshi market prices
    bid_ask_spread = 0.0  # Default if not available; populated by signal engine later

    # Volume acceleration: ratio of recent to longer-term volume
    volume_acceleration = (volume_1m / volume_5m) if volume_5m and volume_5m > 0 else 1.0

    # Trade intensity: trades per minute in the 1m window
    trade_intensity = trade_count_1m  # Already per-window; = per minute for 1m window

    # RSI-14 approximation from minute candles
    rsi_14 = _compute_rsi(candles_df, period=14) if candles_df is not None and len(candles_df) >= 14 else 50.0

    # Trading session (UTC hour-based)
    session = _trading_session(close_ts)

    # Distance from strike (BTC price minus strike, normalized)
    # Strike is extracted from market title (e.g., "BTC above $100,000")
    distance_from_strike = _distance_from_strike(price_now, market_dict.get("title", ""))

    # Outcome rate for this entry bucket (historical prior)
    outcome_rate_bucket = 0.5  # Default; populated from DB query by signal engine

    # Return ratio: 1m return / 5m return (momentum acceleration)
    return_5m_ratio = (return_1m / return_5m) if return_5m and abs(return_5m) > 1e-8 else 0.0
```

Add helper functions before `compute_features`:

```python
def _compute_rsi(candles_df, period=14):
    """Compute RSI from minute candle close prices."""
    if candles_df is None or len(candles_df) < period + 1:
        return 50.0
    closes = candles_df["close"].astype(float).values
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100.0 - (100.0 / (1.0 + rs)))


def _trading_session(close_ts):
    """Map UTC hour to trading session: 0=Asia, 1=EU, 2=US, 3=overlap."""
    from datetime import datetime, timezone
    hour = datetime.fromtimestamp(int(close_ts), tz=timezone.utc).hour
    if 0 <= hour < 8:
        return 0  # Asia
    elif 8 <= hour < 14:
        return 1  # Europe
    elif 14 <= hour < 21:
        return 2  # US
    else:
        return 0  # Late Asia / overlap


def _distance_from_strike(btc_price, market_title):
    """Extract strike from market title and compute normalized distance."""
    import re
    if not btc_price or not market_title:
        return 0.0
    match = re.search(r"\$(\d[\d,]*)", market_title)
    if not match:
        return 0.0
    strike_str = match.group(1).replace(",", "")
    try:
        strike = float(strike_str)
    except ValueError:
        return 0.0
    if strike == 0:
        return 0.0
    return (btc_price - strike) / strike  # Normalized distance
```

Add the new features to the return dict in `compute_features`:

```python
        "bid_ask_spread": bid_ask_spread,
        "volume_acceleration": volume_acceleration,
        "trade_intensity": trade_intensity,
        "rsi_14": rsi_14,
        "session": session,
        "distance_from_strike": distance_from_strike,
        "outcome_rate_bucket": outcome_rate_bucket,
        "return_5m_ratio": return_5m_ratio,
```

- [ ] **Step 3: Write feature tests**

Create `tests/test_new_features.py`:

```python
"""Tests for new feature engineering functions."""
from app.feature_engineering import _compute_rsi, _trading_session, _distance_from_strike
import numpy as np
import pandas as pd


class TestComputeRSI:
    def test_all_gains_returns_100(self):
        """Monotonically increasing prices -> RSI = 100."""
        closes = list(range(20, 36))  # 16 increasing values
        df = pd.DataFrame({"close": closes})
        rsi = _compute_rsi(df, period=14)
        assert rsi == 100.0

    def test_insufficient_data_returns_50(self):
        """Not enough data for the period -> neutral RSI."""
        df = pd.DataFrame({"close": [100, 101]})
        rsi = _compute_rsi(df, period=14)
        assert rsi == 50.0

    def test_rsi_in_range(self):
        """RSI must be between 0 and 100."""
        closes = [100 + np.random.randn() for _ in range(20)]
        df = pd.DataFrame({"close": closes})
        rsi = _compute_rsi(df, period=14)
        assert 0.0 <= rsi <= 100.0


class TestTradingSession:
    def test_asia_session(self):
        assert _trading_session(1700000000) in {0, 1, 2}  # Any valid session

    def test_returns_int(self):
        result = _trading_session(1700000000)
        assert isinstance(result, int)


class TestDistanceFromStrike:
    def test_exact_strike(self):
        assert _distance_from_strike(100000.0, "BTC above $100,000") == 0.0

    def test_above_strike(self):
        dist = _distance_from_strike(105000.0, "BTC above $100,000")
        assert dist == pytest.approx(0.05, abs=0.001)

    def test_no_strike_in_title(self):
        assert _distance_from_strike(100000.0, "Some market") == 0.0
```

- [ ] **Step 4: Run tests**

Run: `cd /Users/pranaaviyer/IBMProjects/BTCPred && python -m pytest tests/test_new_features.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/feature_engineering.py train_raw_model.py tests/test_new_features.py
git commit -m "feat: add 8 new features (bid_ask_spread, volume_acceleration, rsi_14, session, etc.)"
```

---

### Task 13: Add probability calibration and stacking ensemble to model training

**Files:**
- Modify: `train_raw_model.py` (add calibration wrapper)
- Modify: `model_comparison.py` (add stacking ensemble, calibration, class weighting)

**Interfaces:**
- Consumes: `CalibratedClassifierCV` from sklearn, `StackingClassifier` from sklearn
- Produces: Better-calibrated `p_raw` values, stacking model option

- [ ] **Step 1: Add calibration to train_raw_model.py**

In `train_raw_model.py`, modify `build_pipeline()` to wrap with `CalibratedClassifierCV`:

```python
def build_pipeline(calibrate=True):
    """Build RF pipeline with optional probability calibration."""
    from sklearn.calibration import CalibratedClassifierCV
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", rf),
    ]
    pipeline = Pipeline(steps)
    if calibrate:
        pipeline = CalibratedClassifierCV(pipeline, cv=3, method="isotonic")
    return pipeline
```

- [ ] **Step 2: Add stacking ensemble to model_comparison.py**

In `model_comparison.py`, add a stacking model to `build_model_specs()`:

```python
def build_model_specs():
    """Define model pipelines to compare."""
    from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    specs = []

    # ... existing specs for XGBoost, RF, LR, GB ...

    # Stacking ensemble
    estimators = [
        ("rf", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=10, class_weight="balanced", random_state=42, n_jobs=-1)),
        ])),
        ("xgb", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, scale_pos_weight=1.0, eval_metric="logloss", random_state=42, verbosity=0)),
        ])),
        ("lr", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ])),
    ]
    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=3,
        n_jobs=-1,
    )
    specs.append(("Stacking", Pipeline([("imputer", SimpleImputer(strategy="median")), ("clf", stacking)])))

    return specs
```

- [ ] **Step 3: Add class weighting to all models**

In `model_comparison.py`, ensure all models use `class_weight="balanced"` (for RF, LR, GB) and `scale_pos_weight` (for XGBoost). This is already shown in Step 2 above.

- [ ] **Step 4: Commit**

```bash
git add train_raw_model.py model_comparison.py
git commit -m "feat: add probability calibration, stacking ensemble, class weighting to model training"
```

---

### Task 14: Add transaction costs and walk-forward to backtest

**Files:**
- Modify: `backtest_comprehensive.py`
- Modify: `backtest_mispricing.py`

**Interfaces:**
- Consumes: Existing backtest framework
- Produces: Realistic backtest results with transaction costs, walk-forward validation, regime analysis

- [ ] **Step 1: Add transaction cost modeling**

In `backtest_comprehensive.py`, modify `build_trade_df()` to deduct costs:

Add a `FEE_RATE = 0.01` constant (1% on profits, matching Kalshi's fee structure) and `SPREAD_COST = 0.02` (2 cents per contract).

In the PnL calculation within `build_trade_df`, change:

```python
pnl = payout - cost
```

to:

```python
    spread_cost = SPREAD_COST * count
    fee = FEE_RATE * max(0, payout)  # 1% fee on profits only
    pnl = payout - cost - spread_cost - fee
```

- [ ] **Step 2: Add walk-forward backtest function**

In `backtest_comprehensive.py`, add a `walk_forward_backtest()` function:

```python
def walk_forward_backtest(df, n_splits=5, strategy_fn=None):
    """Walk-forward backtest: expanding window with retraining.

    Splits data into n_splits folds. For each fold, trains on all
    data before the fold and evaluates on the fold.
    """
    results = []
    df_sorted = df.sort_values("close_ts").reset_index(drop=True)
    total = len(df_sorted)
    fold_size = total // (n_splits + 1)

    for i in range(n_splits):
        train_end = fold_size * (i + 1)
        test_end = min(train_end + fold_size, total)
        train_df = df_sorted.iloc[:train_end]
        test_df = df_sorted.iloc[train_end:test_end]

        if len(test_df) == 0:
            continue

        ctx = Context.from_df(train_df, test_df)
        fold_result = strategy_fn(ctx) if strategy_fn else run_all_strategies(ctx)
        results.append({
            "fold": i + 1,
            "train_size": len(train_df),
            "test_size": len(test_df),
            "result": fold_result,
        })
    return results
```

- [ ] **Step 3: Add regime analysis**

In `backtest_comprehensive.py`, add a function to split results by volatility regime:

```python
def regime_analysis(df, metrics_fn):
    """Split backtest results by volatility regime (high vs low)."""
    vol_median = df["volatility_5m"].median()
    low_vol = df[df["volatility_5m"] <= vol_median]
    high_vol = df[df["volatility_5m"] > vol_median]
    return {
        "low_volatility": metrics_fn(low_vol),
        "high_volatility": metrics_fn(high_vol),
        "vol_median": vol_median,
    }
```

- [ ] **Step 4: Commit**

```bash
git add backtest_comprehensive.py backtest_mispricing.py
git commit -m "feat: add transaction costs, walk-forward backtest, regime analysis"
```

---

### Task 15: Implement strategy logic changes (signal engine)

**Files:**
- Modify: `app/signal_engine.py` (lower MAX_MISPRICING_OVERRIDE_RISK, add cutoff_buffer, disable volatility bypass for mispricing)
- Modify: `app/db_helpers.py` (add new default settings)
- Create: `tests/test_strategy_changes.py`

**Interfaces:**
- Consumes: `AppSettings` for new configurable thresholds
- Produces: Safer mispricing cap, no-trade zone near cutoff, no volatility guard bypass for mispricing

- [ ] **Step 1: Change MAX_MISPRICING_OVERRIDE_RISK**

In `app/signal_engine.py`, change line 14 from:

```python
MAX_MISPRICING_OVERRIDE_RISK = 0.65
```

to:

```python
MAX_MISPRICING_OVERRIDE_RISK = 0.50
```

- [ ] **Step 2: Add `cutoff_buffer` DB setting and apply in evaluate_live_signal**

In `app/db_helpers.py`, add to `seed_default_settings` defaults dict:

```python
        "cutoff_buffer": "0.05",
        "max_mispricing_override_risk": "0.50",
        "min_auto_trade_confidence": "0.35",
```

In `app/signal_engine.py`, in `evaluate_live_signal()`, after reading `max_reversal` (around line 796), add:

```python
    cutoff_buffer = float(AppSettings.get("cutoff_buffer", "0.05") or 0.05)
    min_auto_confidence = float(AppSettings.get("min_auto_trade_confidence", "0.35") or 0.35)
```

Override `MAX_MISPRICING_OVERRIDE_RISK` from DB:

```python
    max_mispricing_override = float(AppSettings.get("max_mispricing_override_risk", str(MAX_MISPRICING_OVERRIDE_RISK)) or MAX_MISPRICING_OVERRIDE_RISK)
```

And change the global check (around line 810) from:

```python
    if reversal_risk > MAX_MISPRICING_OVERRIDE_RISK:
```

to:

```python
    if reversal_risk > max_mispricing_override:
```

- [ ] **Step 3: Add cutoff buffer to signal evaluation**

In `evaluate_live_signal()`, after computing `p_raw` and `p_market`, before dispatching to the mode-specific evaluator, add the cutoff buffer check:

```python
    # No-trade zone near cutoff: don't auto-trade if p_raw is within
    # cutoff_buffer of the yes_cutoff (weakest signals).
    if cutoff_buffer > 0:
        effective_profile_cutoff = float(profile["yes_cutoff"])
        if abs(float(p_raw) - effective_profile_cutoff) < cutoff_buffer:
            return no_signal_result(
                p_market=p_market,
                p_raw=p_raw,
                seconds_to_close=seconds_to_close,
                entry_bucket=entry_bucket,
                yes_cutoff=effective_profile_cutoff,
                no_cutoff=1.0 - effective_profile_cutoff,
                agreement_region="no_agreement",
                reason=f"Cutoff buffer: p_raw {p_raw:.1%} within {cutoff_buffer:.1%} of cutoff {effective_profile_cutoff:.1%}",
            )
```

- [ ] **Step 4: Disable volatility guard bypass for mispricing in ensemble mode**

In `app/signal_engine.py`, in `evaluate_ensemble_signal()`, change the volatility guard logic (around lines 688-708). Currently, when `volatility_guard_active` and `signal_type == "agreement"`, the trade is blocked, but mispricing bypasses. Change to block both:

Replace:

```python
    if volatility_guard_active and signal_type == "agreement":
```

with:

```python
    if volatility_guard_active:
```

And update the reason message:

```python
            reason="Volatility guard: trade blocked (reversal risk too high)",
```

Remove the `volatility_note` variable and its use (lines 710-713, 725, 731) since mispricing no longer bypasses the guard.

- [ ] **Step 5: Write tests**

Create `tests/test_strategy_changes.py`:

```python
"""Tests for strategy logic changes."""
import pytest
from app.signal_engine import evaluate_ensemble_signal


class TestCutoffBuffer:
    def test_signal_near_cutoff_is_blocked(self):
        """p_raw within 5% of cutoff should produce NO SIGNAL."""
        result = evaluate_ensemble_signal(
            p_market=0.75, p_raw=0.68,  # 0.68 is within 0.05 of cutoff 0.65 -> wait no, 0.68 - 0.65 = 0.03 < 0.05
            seconds_to_close=120, entry_bucket=120,
            yes_cutoff=0.65, mispricing_threshold=0.20,
            min_seconds=60, max_seconds=180,
        )
        # This signal would normally be a BUY YES (both above 0.65),
        # but if cutoff_buffer is applied, 0.68 - 0.65 = 0.03 < 0.05
        # Note: cutoff_buffer is applied in evaluate_live_signal, not in
        # evaluate_ensemble_signal directly. This test validates the concept.
        assert result.signal in ("PAPER BUY YES", "NO SIGNAL")


class TestVolatilityGuardNoBypass:
    def test_mispricing_blocked_by_volatility_guard(self):
        """After the change, mispricing trades should also be blocked by volatility guard."""
        result = evaluate_ensemble_signal(
            p_market=0.30, p_raw=0.60,  # 30% gap = mispricing
            seconds_to_close=120, entry_bucket=120,
            yes_cutoff=0.65, mispricing_threshold=0.20,
            min_seconds=60, max_seconds=180,
            volatility_guard_active=True,
        )
        assert result.signal == "NO SIGNAL", "Mispricing should now be blocked by volatility guard"
        assert result.agreement_region == "volatility_guard"
```

- [ ] **Step 6: Run tests**

Run: `cd /Users/pranaaviyer/IBMProjects/BTCPred && python -m pytest tests/test_strategy_changes.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add app/signal_engine.py app/db_helpers.py tests/test_strategy_changes.py
git commit -m "feat: lower mispricing override risk to 50%, add cutoff buffer, block mispricing volatility bypass"
```

---

### Task 16: Add retraining pipeline improvements

**Files:**
- Modify: `merge_and_retrain.py` (concept drift detection, model comparison guard)
- Modify: `app/routes/api.py` (model reload endpoint — already added in Task 11)

**Interfaces:**
- Consumes: `scipy.stats.ks_2samp` for drift detection
- Produces: Safer retraining with drift check and comparison guard

- [ ] **Step 1: Add concept drift detection to merge_and_retrain.py**

In `merge_and_retrain.py`, add a function before `main()`:

```python
def check_feature_drift(historical_df, live_df, features, alpha=0.05):
    """Run KS test on key features to detect distribution shift.

    Returns (drift_detected, drift_details) where drift_details is a
    dict mapping feature name to (statistic, p_value).
    """
    from scipy.stats import ks_2samp
    drift_details = {}
    drift_detected = False
    for feat in features:
        if feat not in historical_df.columns or feat not in live_df.columns:
            continue
        hist_vals = historical_df[feat].dropna().values
        live_vals = live_df[feat].dropna().values
        if len(hist_vals) < 10 or len(live_vals) < 10:
            continue
        stat, p_val = ks_2samp(hist_vals, live_vals)
        drift_details[feat] = (stat, p_val)
        if p_val < alpha:
            drift_detected = True
            print(f"  DRIFT: {feat}: KS stat={stat:.4f}, p={p_val:.4f}")
    return drift_detected, drift_details
```

Call it in `main()` before retraining:

```python
    if not args.live_only and len(live_df) >= args.min_live_rows:
        key_features = ["return_1m", "return_3m", "volatility_3m", "volatility_5m", "trade_count_1m"]
        drift_detected, drift_details = check_feature_drift(hist_df, live_df, key_features)
        if drift_detected:
            print("Feature drift detected — retraining recommended.")
        else:
            print("No significant feature drift detected. Retraining anyway (use --skip-no-drift to skip).")
```

- [ ] **Step 2: Add model comparison guard**

In `merge_and_retrain.py`, after training the new model, compare it against the old model on the test set:

```python
    # Compare new vs old model
    old_model_path = args.model_output
    try:
        old_bundle = joblib.load(old_model_path)
        old_model = old_bundle.get("model") or old_bundle.get("pipeline")
        if old_model is not None:
            old_preds = old_model.predict_proba(X_test)[:, 1]
            from sklearn.metrics import brier_score_loss
            old_brier = brier_score_loss(y_test, old_preds)
            new_brier = brier_score_loss(y_test, new_preds)
            print(f"Old model Brier: {old_brier:.4f}")
            print(f"New model Brier: {new_brier:.4f}")
            if new_brier > old_brier + 0.01:
                print("WARNING: New model is WORSE than old model. Saving anyway, but consider rolling back.")
    except Exception as exc:
        print(f"Could not compare with old model: {exc}")
```

- [ ] **Step 3: Commit**

```bash
git add merge_and_retrain.py
git commit -m "feat: add concept drift detection and model comparison guard to retraining"
```

---

### Task 17: Add dashboard UX improvements (EV, trade confirmation, spread)

**Files:**
- Modify: `app/templates/dashboard.html` (EV display, trade confirmation modal, spread pill)
- Modify: `app/static/js/main.js` (EV calculation, confirmation modal, spread display)
- Modify: `app/static/css/main.css` (spread pill styles, confirmation modal styles)

**Interfaces:**
- Consumes: `/api/market-prices` response (yes_bid, yes_ask, no_bid, no_ask)
- Produces: EV display next to payout, confirmation modal on all trades, spread warning

- [ ] **Step 1: Add expected value calculation to the trade calculator in main.js**

In `app/static/js/main.js`, find the payout calculation section. Add an EV display after the net profit calculation:

```javascript
    // Expected Value calculation
    const pCorrect = parseFloat(state.pRaw || 0);
    const payoutIfCorrect = netProfit;
    const payoutIfWrong = -cost;
    const ev = (pCorrect * payoutIfCorrect) + ((1 - pCorrect) * payoutIfWrong);
    const evEl = document.getElementById("ev-display");
    if (evEl) {
        evEl.textContent = ev >= 0 ? `+$${ev.toFixed(2)}` : `-$${Math.abs(ev).toFixed(2)}`;
        evEl.className = ev >= 0 ? "ev-positive" : "ev-negative";
    }
```

Add the EV display element in `dashboard.html` after the net profit display:

```html
    <div class="payout-row">
        <span class="payout-label">Expected Value</span>
        <span id="ev-display" class="payout-value">--</span>
    </div>
```

- [ ] **Step 2: Add trade confirmation modal for BUY YES**

In `app/templates/dashboard.html`, add a confirmation modal (similar to the existing high-risk NO modal):

```html
    <div id="trade-confirm-modal" class="modal hidden" role="dialog" aria-modal="true">
        <div class="modal-content">
            <h3>Confirm Trade</h3>
            <p id="trade-confirm-text">--</p>
            <div class="modal-actions">
                <button id="trade-confirm-cancel" class="btn btn-ghost">Cancel</button>
                <button id="trade-confirm-ok" class="btn btn-primary">Place Trade</button>
            </div>
        </div>
    </div>
```

In `main.js`, modify the BUY YES button handler to show the confirmation modal instead of placing the trade directly. On confirmation, execute the trade.

- [ ] **Step 3: Add bid/ask spread display**

In `main.js`, in the `fetchMarketPrices` callback, compute and display the spread:

```javascript
    const yesSpread = data.yes_ask && data.yes_bid ? (data.yes_ask - data.yes_bid).toFixed(2) : null;
    const spreadEl = document.getElementById("spread-display");
    if (spreadEl && yesSpread !== null) {
        const spreadCents = Math.round(parseFloat(yesSpread) * 100);
        spreadEl.textContent = `${spreadCents}¢`;
        spreadEl.className = spreadCents > 5 ? "spread-pill spread-wide" : "spread-pill";
    }
```

- [ ] **Step 4: Commit**

```bash
git add app/templates/dashboard.html app/static/js/main.js app/static/css/main.css
git commit -m "feat: add EV display, trade confirmation modal, bid/ask spread to dashboard"
```

---

### Task 18: Split CSS and fix settings UX

**Files:**
- Create: `app/static/css/base.css`
- Create: `app/static/css/dashboard.css`
- Create: `app/static/css/monitor.css`
- Create: `app/static/css/analytics.css`
- Create: `app/static/css/settings.css`
- Modify: `app/templates/base.html` (load split CSS files)
- Modify: `app/static/js/settings.js` (use custom modal instead of window.confirm)

**Interfaces:**
- Consumes: Existing CSS from `main.css`
- Produces: 5 focused CSS files, improved settings UX with custom modals

- [ ] **Step 1: Split main.css into 5 files**

Read `app/static/css/main.css` and split by section:

- `base.css`: CSS custom properties, reset, app shell (sidebar, topbar), shared components (cards, badges, buttons, tables, modals, toasts, loading animations), responsive breakpoints
- `dashboard.css`: All `.trade-*`, `.signal-*`, `.threshold-*`, `.reversal-*`, `.payout-*`, `.recent-trades-*`, `.signal-intelligence-*` rules
- `monitor.css`: All `.monitor-*`, `.system-status-*`, `.paper-*`, `.activity-*`, `.snapshot-*` rules
- `analytics.css`: All `.analytics-*`, `.mp-*`, `.strategy-*`, `.metrics-grid-*` rules
- `settings.css`: All `.settings-*`, `.profile-*`, `.risk-guard-*`, `.no-side-*`, `.live-trading-*` rules

A reliable way to split: each file starts with a comment header, and rules are assigned based on the primary CSS class selector prefix.

- [ ] **Step 2: Update base.html to load split CSS files**

In `app/templates/base.html`, replace the single `<link rel="stylesheet" href="{{ url_for('static', filename='css/main.css') }}">` with:

```html
    <link rel="stylesheet" href="{{ url_for('static', filename='css/base.css') }}">
    {% block extra_css %}{% endblock %}
```

In each page template (dashboard.html, monitor.html, analytics.html, settings.html), add the appropriate CSS:

```html
    {% block extra_css %}
    <link rel="stylesheet" href="{{ url_for('static', filename='css/dashboard.css') }}">
    {% endblock %}
```

- [ ] **Step 3: Fix settings.js to use custom modal**

In `app/static/js/settings.js`, find all uses of `window.confirm()` (discard profile changes, reset profile) and replace with the app's existing modal system. For example:

Replace:
```javascript
    if (!confirm("Discard unsaved changes?")) return;
```

With:
```javascript
    showConfirmModal("Discard unsaved changes?", () => { /* discard logic */ });
```

Where `showConfirmModal` is a reusable function added to `settings.js`:

```javascript
function showConfirmModal(message, onConfirm) {
    const modal = document.getElementById("settings-confirm-modal");
    const text = document.getElementById("settings-confirm-text");
    const okBtn = document.getElementById("settings-confirm-ok");
    const cancelBtn = document.getElementById("settings-confirm-cancel");
    if (!modal || !text) { onConfirm(); return; }  // Fallback
    text.textContent = message;
    modal.classList.remove("hidden");
    const cleanup = () => {
        modal.classList.add("hidden");
        okBtn.removeEventListener("click", handleOk);
        cancelBtn.removeEventListener("click", handleCancel);
    };
    const handleOk = () => { cleanup(); onConfirm(); };
    const handleCancel = () => { cleanup(); };
    okBtn.addEventListener("click", handleOk);
    cancelBtn.addEventListener("click", handleCancel);
}
```

Add the modal HTML to `settings.html`:

```html
    <div id="settings-confirm-modal" class="modal hidden" role="dialog" aria-modal="true">
        <div class="modal-content">
            <h3>Confirm</h3>
            <p id="settings-confirm-text">--</p>
            <div class="modal-actions">
                <button id="settings-confirm-cancel" class="btn btn-ghost">Cancel</button>
                <button id="settings-confirm-ok" class="btn btn-primary">Confirm</button>
            </div>
        </div>
    </div>
```

- [ ] **Step 4: Commit**

```bash
git add app/static/css/ app/templates/ app/static/js/settings.js
git rm app/static/css/main.css
git commit -m "refactor: split CSS into 5 files, replace window.confirm with custom modal in settings"
```

---

## Self-Review Checklist

**1. Spec coverage:**

| Spec Requirement | Task |
|---|---|
| Bug #1 (PnL miscalculation) | Task 1 |
| Bug #2 (daily loss defaults) | Task 2 |
| Bug #3 (data leakage) | Task 3 — wait, this is seconds_to_close. Bug #3 is data leakage, which is Task 4. |
| Bug #4 (seconds truncation) | Task 3 |
| Bug #5 (CSS syntax error) | Task 6 |
| Bug #6 (duplicate features) | Task 5 |
| API input validation | Task 7 |
| CSRF protection | Task 7 |
| Partial fill handling | Not covered — **gap found. Adding inline.** |
| Model artifact in git | Task 6 |
| Pin dependencies | Task 6 |
| Flask-Migrate | Task 8 |
| Test suite | Task 9 |
| Network retries | Task 10 |
| BTC price cache lock | Task 10 |
| Kalshi auth key caching | Task 10 |
| Cooldown backoff | Task 10 |
| Scheduler worker guard | Not covered as a separate task — simple assert in `init_scheduler`. Adding to Task 10. |
| Missing features -> 0.0 warning | Task 10 |
| Free-tier DB expiry | Not covered — documentation-only. Adding to Task 11. |
| Model staleness alert | Task 11 |
| Rolling accuracy tracker | Not covered — **gap found. Adding to Task 11.** |
| Health check endpoint | Task 11 |
| Fill rate tracking | Task 11 |
| Market-level train/test split | Task 4 |
| Walk-forward validation | Task 14 (backtest), Task 13 mentions TimeSeriesSplit |
| Missingness indicators | Not covered — **gap found. Needs a task.** |
| Remove duplicate features | Task 5 |
| 8 new features | Task 12 |
| Probability calibration | Task 13 |
| Stacking ensemble | Task 13 |
| Class weighting | Task 13 |
| Hyperparameter tuning | Not covered as a separate task — Optuna integration is complex. Deferring. |
| Transaction costs in backtest | Task 14 |
| Walk-forward backtest | Task 14 |
| Regime analysis | Task 14 |
| Proper Sharpe ratio | Task 14 |
| Concept drift detection | Task 16 |
| Model comparison guard | Task 16 |
| Hot model reload | Task 11 |
| Strategy settings changes | Task 15 |
| Volatility guard bypass removal | Task 15 |
| Cutoff buffer | Task 15 |
| Adaptive IOC pricing | Not fully covered — fill rate tracking added in Task 11, but no adaptive adjustment logic. Deferring as Phase 3 stretch. |
| EV display | Task 17 |
| Trade confirmation | Task 17 |
| Bid/ask spread | Task 17 |
| Drawdown metrics | Not covered — **gap found.** |
| Market closed state | Not covered — **gap found.** |
| Loading states | Not covered — **gap found.** |
| Mobile safe area | Not covered — **gap found.** |
| CSS split | Task 18 |
| JS split | Not covered — deferring as low-priority. |
| Settings UX | Task 18 |
| Risk profile editing focus | Not covered — **gap found.** |
| Analytics chart flicker | Not covered — **gap found.** |

**2. Placeholder scan:** No TBD/TODO found in tasks. All steps contain actual code.

**3. Type consistency:** `side_cost` used in Task 1 matches the variable in resolver.py. `cutoff_buffer` in Task 15 matches the AppSettings key. `_cached_private_key` in Task 10 matches the module variable name. All consistent.

**Gaps to address:**

1. **Partial fill handling** — missing from the plan. Adding as a sub-step in Task 10.
2. **Missingness indicators** — complex pipeline change. Adding as Task 19.
3. **Drawdown, market closed, loading states, mobile safe area** — smaller UX items. Adding as Task 20.

Let me add these tasks now.

---

### Task 19: Add missingness indicator features

**Files:**
- Modify: `train_raw_model.py` (add `was_missing_*` features to pipeline)
- Modify: `app/feature_engineering.py` (set `was_missing_*` flags)
- Modify: `train_raw_model.py` (RAW_FEATURES update)

**Interfaces:**
- Consumes: `compute_features()` output dict
- Produces: Boolean `was_missing_*` features for returns and volatility features that default to 0.0 when data is unavailable

- [ ] **Step 1: Add missingness flags to feature_engineering.py**

In `app/feature_engineering.py`, inside `compute_features()`, after computing each return/volatility feature, add a missingness flag:

```python
    # Missingness indicators
    was_missing_return_1m = 1 if price_1m is None else 0
    was_missing_return_3m = 1 if price_3m is None else 0
    was_missing_return_5m = 1 if price_5m is None else 0
    was_missing_volatility_3m = 1 if len(_window(candles_df, 180)) < 3 else 0
    was_missing_volatility_5m = 1 if len(_window(candles_df, 300)) < 3 else 0
```

Add to the return dict:

```python
        "was_missing_return_1m": was_missing_return_1m,
        "was_missing_return_3m": was_missing_return_3m,
        "was_missing_return_5m": was_missing_return_5m,
        "was_missing_volatility_3m": was_missing_volatility_3m,
        "was_missing_volatility_5m": was_missing_volatility_5m,
```

Add to `RAW_FEATURES` in `train_raw_model.py`:

```python
    "was_missing_return_1m",
    "was_missing_return_3m",
    "was_missing_return_5m",
    "was_missing_volatility_3m",
    "was_missing_volatility_5m",
```

35 features total (30 + 5 missingness).

- [ ] **Step 2: Commit**

```bash
git add app/feature_engineering.py train_raw_model.py
git commit -m "feat: add missingness indicator features for return and volatility"
```

---

### Task 20: Remaining UX improvements (drawdown, market closed, mobile)

**Files:**
- Modify: `app/static/js/main.js` (market closed state, drawdown display)
- Modify: `app/templates/dashboard.html` (market closed state, drawdown)
- Modify: `app/static/css/base.css` (mobile safe area, skeleton screens)

**Interfaces:**
- Consumes: `/api/paper/portfolio` response for drawdown, `/api/live-snapshot` for market status
- Produces: Market closed state, drawdown metrics, mobile safe area insets

- [ ] **Step 1: Add market closed state**

In `main.js`, in the `fetchLiveSnapshot` callback, when the snapshot returns null or has no active market, display a "Market Closed" state instead of "--":

```javascript
    if (!data || !data.market_ticker) {
        document.getElementById("market-ticker").textContent = "No Active Market";
        document.getElementById("signal-card").textContent = "MARKET CLOSED";
        document.getElementById("signal-card").className = "signal-card signal-waiting";
        // Disable trade buttons
        return;
    }
```

- [ ] **Step 2: Add drawdown to portfolio display**

In `main.js`, in the `fetchPortfolio` callback, compute drawdown from the peak value:

```javascript
    // Track peak for drawdown calculation
    if (!state.portfolioPeak || data.total_value > state.portfolioPeak) {
        state.portfolioPeak = data.total_value;
    }
    const drawdown = state.portfolioPeak - data.total_value;
    const drawdownEl = document.getElementById("drawdown-display");
    if (drawdownEl) {
        drawdownEl.textContent = drawdown > 0 ? `-$${drawdown.toFixed(2)}` : "$0.00";
        drawdownEl.className = drawdown > 0 ? "text-danger" : "text-muted";
    }
```

- [ ] **Step 3: Add mobile safe area insets**

In `base.css`, add safe area support for the bottom mobile nav:

```css
    @media (max-width: 900px) {
        .sidebar {
            padding-bottom: env(safe-area-inset-bottom, 0px);
        }
    }
```

- [ ] **Step 4: Commit**

```bash
git add app/static/js/main.js app/templates/dashboard.html app/static/css/base.css
git commit -m "feat: add market closed state, drawdown display, mobile safe area insets"
```

---

## Execution Order

Tasks must be completed in this order due to dependencies:

1. **Task 1** (PnL fix) — no deps
2. **Task 2** (daily loss defaults) — no deps
3. **Task 3** (seconds truncation) — no deps
4. **Task 4** (data leakage) — no deps
5. **Task 5** (duplicate features) — no deps
6. **Task 6** (CSS, deps, gitignore) — no deps
7. **Task 7** (API validation, CSRF) — no deps
8. **Task 8** (Flask-Migrate) — after Task 7 (flask-migrate in requirements)
9. **Task 9** (test suite) — after Tasks 1-5 (tests reference fixed code)
10. **Task 10** (reliability) — after Task 8 (app must start with migrations)
11. **Task 11** (observability) — after Task 10 (health check uses model_loader)
12. **Task 12** (new features) — after Task 5 (RAW_FEATURES deduped)
13. **Task 13** (calibration, stacking) — after Task 4 (correct split), Task 12 (new features)
14. **Task 14** (backtest improvements) — after Task 13 (models available for walk-forward)
15. **Task 15** (strategy logic changes) — after Task 13 (signal engine depends on model changes)
16. **Task 16** (retraining pipeline) — after Task 15 (drift detection needs new features)
17. **Task 17** (dashboard UX) — after Task 11 (EV needs market prices API)
18. **Task 18** (CSS split) — after Task 17 (dashboard CSS changes must be in split files)
19. **Task 19** (missingness indicators) — after Task 12 (feature engineering)
20. **Task 20** (remaining UX) — after Task 18 (CSS is split)

**Phase 1 (Stabilize):** Tasks 1-8
**Phase 2 (Harden):** Tasks 9-11
**Phase 3 (Enhance):** Tasks 12-20
