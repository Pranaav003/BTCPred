# BTC-15m Strategy Research & Backtesting Suite — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a modular, path-aware backtesting and strategy-search suite for the Kalshi BTC-15m market that can search across signal rules, path-dependent exits, payoff-aware sizing, and model variants, and crown the best strategy under a robust risk-adjusted objective with hard out-of-sample gates.

**Architecture:** A new `sim/` package. Each market is reconstructed as a `MarketPath` (ordered per-poll price trajectory) so intra-market exits are possible. Pure functions for signals/exits/sizing are composed by a deterministic `engine.simulate()` into a trade ledger; `metrics`, `validation`, and `search` turn ledgers into ranked, gated results; `report` and `promote` are the outputs.

**Tech Stack:** Python 3, pandas, numpy, scikit-learn (already pinned: `scikit-learn==1.9.0`, `joblib==1.4.2`), pytest.

## Global Constraints

- Python 3, `scikit-learn==1.9.0`, `joblib==1.4.2`, pandas, numpy (already in `requirements.txt`).
- One trade per market, entered at the **earliest** qualifying poll (largest `seconds_to_close`). No look-ahead: exit logic may only read polls at or after the entry index.
- Contract prices are in `[0, 1]`. `price_now` in the data is the **YES** contract price. NO mark price = `1 - price_now`.
- Cost model is the **single source of truth** in `sim/costs.py`; no module hardcodes its own spread/fee.
- All randomness is seeded (`seed=42` default) for deterministic tests.
- Every task ends green (`pytest` passing) and is committed.
- Work happens on branch `feature/strategy-search-suite`.

---

### Task 1: Package scaffold + data layer

**Files:**
- Create: `sim/__init__.py`
- Create: `sim/data.py`
- Test: `tests/sim/__init__.py`, `tests/sim/test_data.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `Poll` dataclass: `seconds_to_close: int`, `price_now: float`, `p_raw: float`, `features: dict`
  - `MarketPath` dataclass: `ticker: str`, `bucket: int`, `close_ts: int`, `final_outcome_yes: int`, `polls: list[Poll]` (ordered by **descending** `seconds_to_close`)
  - `load_paths(csv_path: str, min_polls: int = 1) -> list[MarketPath]`

- [ ] **Step 1: Write the failing test**

```python
# tests/sim/test_data.py
import pandas as pd
from sim.data import load_paths, MarketPath, Poll


def _write_csv(tmp_path):
    rows = [
        # market A: two polls, resolves YES
        {"market_ticker": "A", "entry_bucket": 300, "close_ts": 1000,
         "seconds_to_close": 250, "price_now": 0.40, "p_raw": 0.55,
         "return_5m": 10.0, "final_outcome_yes": 1},
        {"market_ticker": "A", "entry_bucket": 300, "close_ts": 1000,
         "seconds_to_close": 120, "price_now": 0.60, "p_raw": 0.58,
         "return_5m": 12.0, "final_outcome_yes": 1},
        # market B: single poll, resolves NO
        {"market_ticker": "B", "entry_bucket": 60, "close_ts": 2000,
         "seconds_to_close": 55, "price_now": 0.30, "p_raw": 0.10,
         "return_5m": -5.0, "final_outcome_yes": 0},
    ]
    p = tmp_path / "d.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return str(p)


def test_load_paths_groups_and_orders(tmp_path):
    paths = load_paths(_write_csv(tmp_path))
    by_ticker = {p.ticker: p for p in paths}
    assert set(by_ticker) == {"A", "B"}

    a = by_ticker["A"]
    assert isinstance(a, MarketPath)
    assert a.bucket == 300 and a.close_ts == 1000 and a.final_outcome_yes == 1
    # ordered by DESCENDING seconds_to_close: earliest (most time left) first
    assert [poll.seconds_to_close for poll in a.polls] == [250, 120]
    assert isinstance(a.polls[0], Poll)
    assert a.polls[0].price_now == 0.40 and a.polls[0].p_raw == 0.55
    assert a.polls[0].features["return_5m"] == 10.0


def test_load_paths_min_polls_filter(tmp_path):
    paths = load_paths(_write_csv(tmp_path), min_polls=2)
    assert {p.ticker for p in paths} == {"A"}  # B has only 1 poll
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/sim/test_data.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sim'`

- [ ] **Step 3: Write minimal implementation**

```python
# sim/__init__.py
```

```python
# tests/sim/__init__.py
```

```python
# sim/data.py
"""Load per-poll Kalshi BTC-15m logs into per-market price paths."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

# Columns that are metadata, not model features.
_META_COLS = {
    "market_ticker", "entry_bucket", "close_ts", "seconds_to_close",
    "price_now", "p_raw", "final_outcome_yes", "logged_at", "signal",
    "agreement_region", "source",
}


@dataclass
class Poll:
    seconds_to_close: int
    price_now: float
    p_raw: float
    features: dict = field(default_factory=dict)


@dataclass
class MarketPath:
    ticker: str
    bucket: int
    close_ts: int
    final_outcome_yes: int
    polls: list  # list[Poll], ordered by DESCENDING seconds_to_close


def load_paths(csv_path: str, min_polls: int = 1) -> list:
    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c not in _META_COLS]
    paths: list = []
    for (ticker, bucket), grp in df.groupby(["market_ticker", "entry_bucket"]):
        grp = grp.sort_values("seconds_to_close", ascending=False)
        if len(grp) < min_polls:
            continue
        polls = [
            Poll(
                seconds_to_close=int(r.seconds_to_close),
                price_now=float(r.price_now),
                p_raw=float(r.p_raw),
                features={c: float(getattr(r, c)) for c in feature_cols
                          if pd.notna(getattr(r, c))},
            )
            for r in grp.itertuples(index=False)
        ]
        first = grp.iloc[0]
        paths.append(MarketPath(
            ticker=str(ticker),
            bucket=int(bucket),
            close_ts=int(first["close_ts"]),
            final_outcome_yes=int(first["final_outcome_yes"]),
            polls=polls,
        ))
    return paths
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/sim/test_data.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add sim/__init__.py sim/data.py tests/sim/__init__.py tests/sim/test_data.py
git commit -m "feat(sim): data layer — load per-poll logs into MarketPath objects"
```

---

### Task 2: Cost model (single source of truth)

**Files:**
- Create: `sim/costs.py`
- Test: `tests/sim/test_costs.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `CostModel` dataclass: `spread_offset_yes=0.02`, `spread_offset_no_low=0.05`, `spread_offset_no_high=0.03`, `no_low_threshold=0.40`, `exit_spread=0.02`, `fee_rate=0.01`, `max_price=0.99`
  - `mark_price(side: str, price_now: float) -> float`
  - `entry_cost(side: str, price_now: float, cfg: CostModel) -> float`
  - `exit_proceeds(side: str, price_now: float, cfg: CostModel) -> float`
  - `fee(gross_gain: float, cfg: CostModel) -> float`

- [ ] **Step 1: Write the failing test**

```python
# tests/sim/test_costs.py
import pytest
from sim.costs import CostModel, mark_price, entry_cost, exit_proceeds, fee


def test_mark_price_sides():
    assert mark_price("yes", 0.40) == pytest.approx(0.40)
    assert mark_price("no", 0.40) == pytest.approx(0.60)


def test_entry_cost_adds_offsets():
    cfg = CostModel()
    # YES: mark 0.40 + 0.02
    assert entry_cost("yes", 0.40, cfg) == pytest.approx(0.42)
    # NO cheap: mark = 1-0.70 = 0.30 (<=0.40) -> +0.05
    assert entry_cost("no", 0.70, cfg) == pytest.approx(0.35)
    # NO expensive: mark = 1-0.30 = 0.70 (>0.40) -> +0.03
    assert entry_cost("no", 0.30, cfg) == pytest.approx(0.73)
    # capped at max_price
    assert entry_cost("yes", 0.99, cfg) == pytest.approx(0.99)


def test_exit_proceeds_subtracts_spread_and_floors_at_zero():
    cfg = CostModel()
    assert exit_proceeds("yes", 0.50, cfg) == pytest.approx(0.48)
    assert exit_proceeds("no", 0.90, cfg) == pytest.approx(0.08)  # mark .10 - .02
    assert exit_proceeds("yes", 0.01, cfg) == pytest.approx(0.0)  # floored


def test_fee_only_on_gains():
    cfg = CostModel()
    assert fee(10.0, cfg) == pytest.approx(0.10)
    assert fee(-10.0, cfg) == pytest.approx(0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/sim/test_costs.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sim.costs'`

- [ ] **Step 3: Write minimal implementation**

```python
# sim/costs.py
"""Single source of truth for entry/exit costs and fees on Kalshi contracts."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CostModel:
    spread_offset_yes: float = 0.02
    spread_offset_no_low: float = 0.05
    spread_offset_no_high: float = 0.03
    no_low_threshold: float = 0.40
    exit_spread: float = 0.02
    fee_rate: float = 0.01
    max_price: float = 0.99


def mark_price(side: str, price_now: float) -> float:
    """Current contract mark for the given side (0-1)."""
    return price_now if side == "yes" else 1.0 - price_now


def entry_cost(side: str, price_now: float, cfg: CostModel) -> float:
    """Per-contract cost to enter, incl. aggressive-fill offset, capped."""
    mark = mark_price(side, price_now)
    if side == "yes":
        offset = cfg.spread_offset_yes
    else:
        offset = (cfg.spread_offset_no_low if mark <= cfg.no_low_threshold
                  else cfg.spread_offset_no_high)
    return min(cfg.max_price, mark + offset)


def exit_proceeds(side: str, price_now: float, cfg: CostModel) -> float:
    """Per-contract proceeds from selling at the current mark, net of spread."""
    return max(0.0, mark_price(side, price_now) - cfg.exit_spread)


def fee(gross_gain: float, cfg: CostModel) -> float:
    """Fee applied only to positive gross gains."""
    return gross_gain * cfg.fee_rate if gross_gain > 0 else 0.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/sim/test_costs.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add sim/costs.py tests/sim/test_costs.py
git commit -m "feat(sim): cost model — single source of truth for entry/exit/fees"
```

---

### Task 3: Metrics

**Files:**
- Create: `sim/metrics.py`
- Test: `tests/sim/test_metrics.py`

**Interfaces:**
- Consumes: nothing (operates on plain lists of pnl floats).
- Produces:
  - `breakeven_win_rate(avg_win: float, avg_loss: float) -> float` (`avg_loss` passed as a positive magnitude)
  - `compute_metrics(pnls: list[float], contracts: list[int]) -> dict` with keys: `n_trades`, `win_rate`, `total_pnl`, `avg_win`, `avg_loss`, `profit_factor`, `sharpe`, `sortino`, `max_drawdown`, `ev_per_contract`
  - `calibration(probs: list[float], outcomes: list[int]) -> dict` with keys: `brier`, `log_loss`

- [ ] **Step 1: Write the failing test**

```python
# tests/sim/test_metrics.py
import math
import pytest
from sim.metrics import breakeven_win_rate, compute_metrics, calibration


def test_breakeven_win_rate():
    # avg win 0.30, avg loss 0.62 -> 0.62 / 0.92
    assert breakeven_win_rate(0.30, 0.62) == pytest.approx(0.62 / 0.92)


def test_compute_metrics_basic():
    pnls = [1.0, 1.0, -2.0, 1.0]  # 3 wins, 1 loss
    contracts = [1, 1, 1, 1]
    m = compute_metrics(pnls, contracts)
    assert m["n_trades"] == 4
    assert m["win_rate"] == pytest.approx(0.75)
    assert m["total_pnl"] == pytest.approx(1.0)
    assert m["avg_win"] == pytest.approx(1.0)
    assert m["avg_loss"] == pytest.approx(-2.0)
    assert m["profit_factor"] == pytest.approx(3.0 / 2.0)
    assert m["ev_per_contract"] == pytest.approx(1.0 / 4)
    # max drawdown: equity 1,2,0,1 -> peak 2 then 0 => dd 2.0
    assert m["max_drawdown"] == pytest.approx(2.0)


def test_compute_metrics_empty():
    m = compute_metrics([], [])
    assert m["n_trades"] == 0
    assert m["total_pnl"] == 0.0
    assert m["profit_factor"] == 0.0


def test_calibration_brier():
    # perfect predictions -> brier 0
    assert calibration([1.0, 0.0], [1, 0])["brier"] == pytest.approx(0.0)
    # coin-flip on a certain event
    assert calibration([0.5, 0.5], [1, 0])["brier"] == pytest.approx(0.25)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/sim/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sim.metrics'`

- [ ] **Step 3: Write minimal implementation**

```python
# sim/metrics.py
"""P&L, risk, and calibration metrics computed from a trade ledger."""
from __future__ import annotations

import math


def breakeven_win_rate(avg_win: float, avg_loss: float) -> float:
    """Win rate needed to break even. avg_loss is a positive magnitude."""
    denom = avg_win + avg_loss
    return avg_loss / denom if denom > 0 else 0.0


def _std(xs: list) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mean = sum(xs) / n
    return math.sqrt(sum((x - mean) ** 2 for x in xs) / (n - 1))


def compute_metrics(pnls: list, contracts: list) -> dict:
    n = len(pnls)
    if n == 0:
        return {
            "n_trades": 0, "win_rate": 0.0, "total_pnl": 0.0, "avg_win": 0.0,
            "avg_loss": 0.0, "profit_factor": 0.0, "sharpe": 0.0,
            "sortino": 0.0, "max_drawdown": 0.0, "ev_per_contract": 0.0,
        }
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gross_win = sum(wins)
    gross_loss = -sum(losses)
    total = sum(pnls)

    # equity curve max drawdown
    equity, peak, max_dd = 0.0, 0.0, 0.0
    for p in pnls:
        equity += p
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)

    downside = [p for p in pnls if p < 0]
    total_contracts = sum(contracts) if contracts else 0
    return {
        "n_trades": n,
        "win_rate": len(wins) / n,
        "total_pnl": total,
        "avg_win": (gross_win / len(wins)) if wins else 0.0,
        "avg_loss": (sum(losses) / len(losses)) if losses else 0.0,
        "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else 0.0,
        "sharpe": (total / n) / _std(pnls) if _std(pnls) > 0 else 0.0,
        "sortino": (total / n) / _std(downside) if _std(downside) > 0 else 0.0,
        "max_drawdown": max_dd,
        "ev_per_contract": (total / total_contracts) if total_contracts else 0.0,
    }


def calibration(probs: list, outcomes: list) -> dict:
    n = len(probs)
    if n == 0:
        return {"brier": 0.0, "log_loss": 0.0}
    brier = sum((p - y) ** 2 for p, y in zip(probs, outcomes)) / n
    eps = 1e-12
    ll = -sum(
        y * math.log(min(max(p, eps), 1 - eps))
        + (1 - y) * math.log(min(max(1 - p, eps), 1 - eps))
        for p, y in zip(probs, outcomes)
    ) / n
    return {"brier": brier, "log_loss": ll}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/sim/test_metrics.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add sim/metrics.py tests/sim/test_metrics.py
git commit -m "feat(sim): metrics — earning, risk, and calibration measures"
```

---

### Task 4: Entry signals

**Files:**
- Create: `sim/signals.py`
- Test: `tests/sim/test_signals.py`

**Interfaces:**
- Consumes: `MarketPath`, `Poll` from `sim.data`.
- Produces:
  - `EntryDecision` dataclass: `entry_idx: int`, `side: str`
  - `ensemble_signal(path: MarketPath, cfg: dict) -> EntryDecision | None`
  - `mean_reversion_signal(path: MarketPath, cfg: dict) -> EntryDecision | None`
  - `SIGNALS: dict[str, callable]` registry mapping `"ensemble"` and `"mean_reversion"` to the functions above
  - Signal `cfg` keys read: `mispricing_threshold`, `yes_cutoff`, `max_entry_yes`, `max_entry_no`, `no_max_p_raw`, `cutoff_buffer`, `min_entry_price`, `min_seconds`, `max_seconds`, and (mean-reversion) `mr_return_5m`, `mr_price_floor`.

- [ ] **Step 1: Write the failing test**

```python
# tests/sim/test_signals.py
from sim.data import MarketPath, Poll
from sim.signals import ensemble_signal, mean_reversion_signal, EntryDecision, SIGNALS


def _cfg(**over):
    base = dict(mispricing_threshold=0.25, yes_cutoff=0.72, max_entry_yes=0.65,
                max_entry_no=0.80, no_max_p_raw=0.20, cutoff_buffer=0.03,
                min_entry_price=0.05, min_seconds=60, max_seconds=300,
                mr_return_5m=40.0, mr_price_floor=0.60)
    base.update(over)
    return base


def test_ensemble_bearish_gap_triggers_no():
    # market 29%, model 12% -> bearish gap 0.17 >= 0.25? no. Use bigger gap.
    path = MarketPath("A", 300, 1000, 0, [
        Poll(200, 0.40, 0.12, {}),  # gap = 0.12-0.40 = -0.28 -> bearish; p_raw<0.20 ok
    ])
    d = ensemble_signal(path, _cfg())
    assert isinstance(d, EntryDecision)
    assert d.side == "no" and d.entry_idx == 0


def test_ensemble_no_blocked_when_praw_too_high():
    path = MarketPath("A", 300, 1000, 0, [
        Poll(200, 0.70, 0.35, {}),  # bearish gap 0.35 but p_raw 0.35 >= no_max_p_raw
    ])
    assert ensemble_signal(path, _cfg()) is None


def test_ensemble_bullish_mispricing_triggers_yes():
    path = MarketPath("A", 300, 1000, 1, [
        Poll(200, 0.30, 0.60, {}),  # gap +0.30 >= 0.25 and p_raw>=0.5, entry 0.30 <= 0.65
    ])
    d = ensemble_signal(path, _cfg())
    assert d.side == "yes" and d.entry_idx == 0


def test_ensemble_respects_time_window_and_picks_earliest():
    path = MarketPath("A", 300, 1000, 0, [
        Poll(400, 0.40, 0.10, {}),  # outside window (>300)
        Poll(250, 0.40, 0.10, {}),  # first in-window bearish -> pick this
        Poll(120, 0.40, 0.10, {}),
    ])
    d = ensemble_signal(path, _cfg())
    assert d.entry_idx == 1 and d.side == "no"


def test_mean_reversion_fades_big_up_move():
    # large positive return_5m + expensive YES -> fade with NO
    path = MarketPath("A", 300, 1000, 0, [
        Poll(200, 0.70, 0.50, {"return_5m": 55.0}),
    ])
    d = mean_reversion_signal(path, _cfg())
    assert d.side == "no" and d.entry_idx == 0


def test_registry():
    assert SIGNALS["ensemble"] is ensemble_signal
    assert SIGNALS["mean_reversion"] is mean_reversion_signal
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/sim/test_signals.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sim.signals'`

- [ ] **Step 3: Write minimal implementation**

```python
# sim/signals.py
"""Pluggable entry-signal functions. Each returns the earliest qualifying entry."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EntryDecision:
    entry_idx: int
    side: str  # "yes" or "no"


def _in_window(seconds: int, cfg: dict) -> bool:
    return cfg["min_seconds"] <= seconds <= cfg["max_seconds"]


def ensemble_signal(path, cfg: dict):
    """Vectorized port of evaluate_ensemble_signal, scanning earliest-first."""
    thresh = cfg["mispricing_threshold"]
    yes_cut = cfg["yes_cutoff"]
    for idx, poll in enumerate(path.polls):
        if not _in_window(poll.seconds_to_close, cfg):
            continue
        p_market, p_raw = poll.price_now, poll.p_raw
        if abs(p_raw - yes_cut) < cfg["cutoff_buffer"]:
            continue  # noisy edge zone
        gap = p_raw - p_market
        agreement_yes = p_market >= yes_cut and p_raw >= yes_cut
        mispricing_bull = gap >= thresh and p_raw >= 0.50
        mispricing_bear = (-gap) >= thresh and p_raw < 0.50
        yes_ok = cfg["min_entry_price"] <= p_market <= cfg["max_entry_yes"]
        no_ok = cfg["min_entry_price"] <= (1.0 - p_market) <= cfg["max_entry_no"]
        no_praw_ok = p_raw < cfg["no_max_p_raw"]
        if (agreement_yes or mispricing_bull) and yes_ok:
            return EntryDecision(idx, "yes")
        if mispricing_bear and no_ok and no_praw_ok:
            return EntryDecision(idx, "no")
    return None


def mean_reversion_signal(path, cfg: dict):
    """Fade a large recent BTC move: big up-move + pricey YES -> buy NO, and vice-versa."""
    for idx, poll in enumerate(path.polls):
        if not _in_window(poll.seconds_to_close, cfg):
            continue
        r5 = poll.features.get("return_5m", 0.0)
        p_market = poll.price_now
        if r5 >= cfg["mr_return_5m"] and p_market >= cfg["mr_price_floor"]:
            return EntryDecision(idx, "no")
        if r5 <= -cfg["mr_return_5m"] and p_market <= (1.0 - cfg["mr_price_floor"]):
            return EntryDecision(idx, "yes")
    return None


SIGNALS = {
    "ensemble": ensemble_signal,
    "mean_reversion": mean_reversion_signal,
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/sim/test_signals.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add sim/signals.py tests/sim/test_signals.py
git commit -m "feat(sim): entry signals — ensemble port + mean-reversion family"
```

---

### Task 5: Path-dependent exits

**Files:**
- Create: `sim/exits.py`
- Test: `tests/sim/test_exits.py`

**Interfaces:**
- Consumes: `MarketPath`, `Poll` from `sim.data`; `mark_price` from `sim.costs`.
- Produces:
  - `ExitResult` dataclass: `exit_idx: int`, `exit_price: float | None` (None = held to resolution), `reason: str`
  - `hold_to_resolution(path, entry_idx, side, cfg) -> ExitResult`
  - `take_profit_stop_loss(path, entry_idx, side, cfg) -> ExitResult`
  - `trailing_stop(path, entry_idx, side, cfg) -> ExitResult`
  - `EXITS: dict[str, callable]` registry: `"hold"`, `"tp_sl"`, `"trailing"`
  - Exit `cfg` keys read: `tp_abs` (0 = disabled), `sl_abs` (0 = disabled), `trail_abs`.

- [ ] **Step 1: Write the failing test**

```python
# tests/sim/test_exits.py
from sim.data import MarketPath, Poll
from sim.exits import (hold_to_resolution, take_profit_stop_loss, trailing_stop,
                       ExitResult, EXITS)


def _path(prices):
    polls = [Poll(300 - i * 30, p, 0.5, {}) for i, p in enumerate(prices)]
    return MarketPath("A", 300, 1000, 1, polls)


def test_hold_to_resolution():
    r = hold_to_resolution(_path([0.4, 0.5, 0.6]), 0, "yes", {})
    assert r.exit_price is None and r.reason == "resolution"
    assert r.exit_idx == 2  # last poll index


def test_stop_loss_triggers_for_yes():
    # YES entered at mark 0.50; price falls; SL 0.10 -> exit when mark <= 0.40
    r = take_profit_stop_loss(_path([0.50, 0.45, 0.38, 0.30]), 0, "yes",
                              {"tp_abs": 0.0, "sl_abs": 0.10})
    assert r.reason == "stop_loss" and r.exit_idx == 2
    assert r.exit_price == 0.38


def test_take_profit_triggers_for_no():
    # NO entered at price_now 0.50 -> mark 0.50; NO mark rises as price_now FALLS.
    # price_now 0.50 -> 0.30 => NO mark 0.50 -> 0.70, gain +0.20 >= tp 0.15
    r = take_profit_stop_loss(_path([0.50, 0.40, 0.30]), 0, "no",
                              {"tp_abs": 0.15, "sl_abs": 0.0})
    assert r.reason == "take_profit" and r.exit_idx == 2
    assert r.exit_price == 0.30


def test_no_trigger_holds():
    r = take_profit_stop_loss(_path([0.50, 0.52, 0.51]), 0, "yes",
                              {"tp_abs": 0.20, "sl_abs": 0.20})
    assert r.reason == "resolution" and r.exit_price is None


def test_trailing_stop_locks_gain():
    # YES entry 0.50, rises to 0.70 (peak), trail 0.10 -> exit when mark <= 0.60
    r = trailing_stop(_path([0.50, 0.70, 0.58]), 0, "yes", {"trail_abs": 0.10})
    assert r.reason == "trailing" and r.exit_idx == 2 and r.exit_price == 0.58


def test_registry():
    assert EXITS["hold"] is hold_to_resolution
    assert EXITS["tp_sl"] is take_profit_stop_loss
    assert EXITS["trailing"] is trailing_stop
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/sim/test_exits.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sim.exits'`

- [ ] **Step 3: Write minimal implementation**

```python
# sim/exits.py
"""Pluggable exit policies operating on a market's intra-market price path.

All comparisons are in *position mark* terms (NO mark = 1 - price_now), so a
gain is favorable regardless of side. Exits may only read polls at or after the
entry index (no look-ahead).
"""
from __future__ import annotations

from dataclasses import dataclass

from sim.costs import mark_price


@dataclass
class ExitResult:
    exit_idx: int
    exit_price: float | None  # None => held to resolution (use final outcome)
    reason: str


def hold_to_resolution(path, entry_idx: int, side: str, cfg: dict) -> ExitResult:
    return ExitResult(exit_idx=len(path.polls) - 1, exit_price=None,
                      reason="resolution")


def take_profit_stop_loss(path, entry_idx: int, side: str, cfg: dict) -> ExitResult:
    tp = cfg.get("tp_abs", 0.0)
    sl = cfg.get("sl_abs", 0.0)
    entry_mark = mark_price(side, path.polls[entry_idx].price_now)
    for idx in range(entry_idx + 1, len(path.polls)):
        poll = path.polls[idx]
        gain = mark_price(side, poll.price_now) - entry_mark
        if tp > 0 and gain >= tp:
            return ExitResult(idx, poll.price_now, "take_profit")
        if sl > 0 and gain <= -sl:
            return ExitResult(idx, poll.price_now, "stop_loss")
    return ExitResult(len(path.polls) - 1, None, "resolution")


def trailing_stop(path, entry_idx: int, side: str, cfg: dict) -> ExitResult:
    trail = cfg.get("trail_abs", 0.0)
    peak = mark_price(side, path.polls[entry_idx].price_now)
    for idx in range(entry_idx + 1, len(path.polls)):
        poll = path.polls[idx]
        mark = mark_price(side, poll.price_now)
        peak = max(peak, mark)
        if trail > 0 and mark <= peak - trail:
            return ExitResult(idx, poll.price_now, "trailing")
    return ExitResult(len(path.polls) - 1, None, "resolution")


EXITS = {
    "hold": hold_to_resolution,
    "tp_sl": take_profit_stop_loss,
    "trailing": trailing_stop,
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/sim/test_exits.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add sim/exits.py tests/sim/test_exits.py
git commit -m "feat(sim): path-dependent exits — hold, TP/SL, trailing stop"
```

---

### Task 6: Payoff-aware sizing

**Files:**
- Create: `sim/sizing.py`
- Test: `tests/sim/test_sizing.py`

**Interfaces:**
- Consumes: `breakeven_win_rate` from `sim.metrics`.
- Produces:
  - `flat_size(edge, win_prob, entry_cost, cfg) -> float` (dollars)
  - `fractional_kelly_size(edge, win_prob, entry_cost, cfg) -> float`
  - `payoff_aware_size(edge, win_prob, entry_cost, cfg) -> float` (returns 0.0 to skip when the trade cannot clear its breakeven win rate)
  - `SIZERS: dict[str, callable]`: `"flat"`, `"kelly"`, `"payoff_aware"`
  - Sizing `cfg` keys read: `base_size`, `kelly_fraction`, `max_size`.

- [ ] **Step 1: Write the failing test**

```python
# tests/sim/test_sizing.py
import pytest
from sim.sizing import flat_size, fractional_kelly_size, payoff_aware_size, SIZERS


def _cfg(**o):
    b = dict(base_size=5.0, kelly_fraction=0.5, max_size=20.0)
    b.update(o)
    return b


def test_flat_size_is_constant():
    assert flat_size(0.1, 0.6, 0.42, _cfg()) == 5.0


def test_kelly_scales_with_edge_and_caps():
    # win_prob 0.7, entry_cost 0.5 -> payoff b = (1-0.5)/0.5 = 1.0
    # kelly f = (b*p - (1-p))/b = (0.7 - 0.3)/1 = 0.4 ; half-kelly -> 0.2 of base*...
    size = fractional_kelly_size(0.2, 0.7, 0.5, _cfg())
    assert size > 0
    # never exceeds max_size
    assert fractional_kelly_size(0.9, 0.99, 0.01, _cfg()) <= 20.0


def test_kelly_zero_when_no_edge():
    # win_prob below breakeven -> non-positive kelly -> 0
    assert fractional_kelly_size(0.0, 0.30, 0.5, _cfg()) == 0.0


def test_payoff_aware_skips_below_breakeven():
    # entry_cost 0.78 -> if lose, lose 0.78; if win, gain 0.22. breakeven wr = .78
    # win_prob 0.60 < breakeven -> skip (0.0)
    assert payoff_aware_size(0.1, 0.60, 0.78, _cfg()) == 0.0
    # win_prob 0.85 > breakeven -> positive size
    assert payoff_aware_size(0.1, 0.85, 0.78, _cfg()) > 0.0


def test_registry():
    assert set(SIZERS) == {"flat", "kelly", "payoff_aware"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/sim/test_sizing.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sim.sizing'`

- [ ] **Step 3: Write minimal implementation**

```python
# sim/sizing.py
"""Position sizing functions. All return a dollar stake (0.0 = skip the trade).

`entry_cost` is the per-contract cost (0-1). Payoff if win = (1 - entry_cost)
per contract; loss if lose = entry_cost per contract.
"""
from __future__ import annotations

from sim.metrics import breakeven_win_rate


def flat_size(edge: float, win_prob: float, entry_cost: float, cfg: dict) -> float:
    return cfg["base_size"]


def fractional_kelly_size(edge: float, win_prob: float, entry_cost: float,
                          cfg: dict) -> float:
    if entry_cost <= 0 or entry_cost >= 1:
        return 0.0
    b = (1.0 - entry_cost) / entry_cost  # net odds per unit staked
    p = win_prob
    kelly = (b * p - (1.0 - p)) / b
    if kelly <= 0:
        return 0.0
    stake = cfg["base_size"] * cfg["kelly_fraction"] * kelly * 10.0
    return min(stake, cfg["max_size"])


def payoff_aware_size(edge: float, win_prob: float, entry_cost: float,
                      cfg: dict) -> float:
    avg_win = 1.0 - entry_cost
    avg_loss = entry_cost
    if win_prob <= breakeven_win_rate(avg_win, avg_loss):
        return 0.0
    # scale base by how far above breakeven, capped
    margin = win_prob - breakeven_win_rate(avg_win, avg_loss)
    return min(cfg["base_size"] * (1.0 + margin * 5.0), cfg["max_size"])


SIZERS = {
    "flat": flat_size,
    "kelly": fractional_kelly_size,
    "payoff_aware": payoff_aware_size,
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/sim/test_sizing.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add sim/sizing.py tests/sim/test_sizing.py
git commit -m "feat(sim): payoff-aware sizing — flat, fractional-Kelly, breakeven-gated"
```

---

### Task 7: Simulation engine

**Files:**
- Create: `sim/engine.py`
- Test: `tests/sim/test_engine.py`

**Interfaces:**
- Consumes: `MarketPath` (`sim.data`); `EntryDecision` (`sim.signals`); `ExitResult` (`sim.exits`); `CostModel`, `mark_price`, `entry_cost`, `exit_proceeds`, `fee` (`sim.costs`).
- Produces:
  - `Trade` dataclass: `ticker: str`, `side: str`, `contracts: int`, `entry_cost_per: float`, `implied_prob: float`, `pnl: float`, `won: bool`, `exit_reason: str`
  - `simulate(paths, signal_fn, exit_fn, sizing_fn, cfg, cost_model) -> list[Trade]`
  - `signal_fn(path, cfg) -> EntryDecision | None`; `exit_fn(path, entry_idx, side, cfg) -> ExitResult`; `sizing_fn(edge, win_prob, entry_cost, cfg) -> float`.
  - Engine passes `win_prob = p_raw` for YES entries and `1 - p_raw` for NO entries to the sizer; `edge = abs(p_raw - p_market)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/sim/test_engine.py
import pytest
from sim.data import MarketPath, Poll
from sim.costs import CostModel
from sim.engine import simulate, Trade
from sim.signals import ensemble_signal
from sim.exits import hold_to_resolution, take_profit_stop_loss
from sim.sizing import flat_size


def _cfg(**o):
    b = dict(mispricing_threshold=0.25, yes_cutoff=0.72, max_entry_yes=0.65,
             max_entry_no=0.80, no_max_p_raw=0.20, cutoff_buffer=0.03,
             min_entry_price=0.05, min_seconds=60, max_seconds=300,
             mr_return_5m=40.0, mr_price_floor=0.60,
             base_size=1.0, kelly_fraction=0.5, max_size=20.0,
             tp_abs=0.0, sl_abs=0.0, trail_abs=0.0)
    b.update(o)
    return b


def test_no_trade_when_signal_none():
    path = MarketPath("A", 300, 1000, 1, [Poll(200, 0.50, 0.50, {})])  # no signal
    trades = simulate([path], ensemble_signal, hold_to_resolution, flat_size,
                      _cfg(), CostModel())
    assert trades == []


def test_winning_no_held_to_resolution():
    # bearish gap NO, market resolves NO (final_outcome_yes=0) -> win
    path = MarketPath("A", 300, 1000, 0, [Poll(200, 0.40, 0.10, {})])
    trades = simulate([path], ensemble_signal, hold_to_resolution, flat_size,
                      _cfg(), CostModel())
    assert len(trades) == 1
    t = trades[0]
    assert isinstance(t, Trade) and t.side == "no" and t.won is True
    # entry cost per: NO mark = 0.60 (>0.40) -> +0.03 = 0.63; contracts=int(1/0.63)=1
    assert t.contracts == 1
    assert t.entry_cost_per == pytest.approx(0.63)
    # proceeds 1.0, gross = 1.0-0.63 = 0.37, fee 1% of 0.37 -> pnl 0.3663
    assert t.pnl == pytest.approx(0.37 * 0.99, abs=1e-6)
    assert t.implied_prob == pytest.approx(0.60)


def test_losing_trade_full_loss():
    # bearish NO but market resolves YES -> loss of full entry cost
    path = MarketPath("A", 300, 1000, 1, [Poll(200, 0.40, 0.10, {})])
    trades = simulate([path], ensemble_signal, hold_to_resolution, flat_size,
                      _cfg(), CostModel())
    t = trades[0]
    assert t.won is False
    assert t.pnl == pytest.approx(-0.63)  # 1 contract * -entry_cost


def test_stop_loss_caps_loss_smaller_than_full():
    # NO entered mark 0.60; price_now rises 0.40->0.55 => NO mark 0.60->0.45,
    # gain -0.15 <= -sl(0.10) -> exit early; resolves YES so hold would be full loss.
    path = MarketPath("A", 300, 1000, 1, [
        Poll(200, 0.40, 0.10, {}),
        Poll(150, 0.55, 0.10, {}),
    ])
    trades = simulate([path], ensemble_signal, take_profit_stop_loss, flat_size,
                      _cfg(sl_abs=0.10), CostModel())
    t = trades[0]
    assert t.exit_reason == "stop_loss"
    # exit proceeds: NO mark at 0.55 = 0.45, minus exit spread 0.02 = 0.43
    # gross = 0.43 - 0.63 = -0.20 -> pnl -0.20 (smaller than full -0.63)
    assert t.pnl == pytest.approx(-0.20)
    assert t.pnl > -0.63
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/sim/test_engine.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sim.engine'`

- [ ] **Step 3: Write minimal implementation**

```python
# sim/engine.py
"""Deterministic backtest engine: compose signal + exit + sizing into trades."""
from __future__ import annotations

from dataclasses import dataclass

from sim.costs import CostModel, mark_price, entry_cost, exit_proceeds, fee


@dataclass
class Trade:
    ticker: str
    side: str
    contracts: int
    entry_cost_per: float
    implied_prob: float
    pnl: float
    won: bool
    exit_reason: str


def simulate(paths, signal_fn, exit_fn, sizing_fn, cfg: dict,
             cost_model: CostModel) -> list:
    trades: list = []
    for path in paths:
        decision = signal_fn(path, cfg)
        if decision is None:
            continue
        side = decision.side
        entry_poll = path.polls[decision.entry_idx]
        e_cost = entry_cost(side, entry_poll.price_now, cost_model)
        implied = mark_price(side, entry_poll.price_now)

        win_prob = entry_poll.p_raw if side == "yes" else 1.0 - entry_poll.p_raw
        edge = abs(entry_poll.p_raw - entry_poll.price_now)
        stake = sizing_fn(edge, win_prob, e_cost, cfg)
        contracts = int(stake / e_cost) if e_cost > 0 else 0
        if contracts <= 0:
            continue

        result = exit_fn(path, decision.entry_idx, side, cfg)
        if result.exit_price is None:  # held to resolution
            won_outcome = (side == "yes" and path.final_outcome_yes == 1) or \
                          (side == "no" and path.final_outcome_yes == 0)
            proceeds_per = 1.0 if won_outcome else 0.0
        else:
            proceeds_per = exit_proceeds(side, result.exit_price, cost_model)

        gross = (proceeds_per - e_cost) * contracts
        pnl = gross - fee(gross, cost_model)
        trades.append(Trade(
            ticker=path.ticker, side=side, contracts=contracts,
            entry_cost_per=e_cost, implied_prob=implied,
            pnl=pnl, won=pnl > 0, exit_reason=result.reason,
        ))
    return trades
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/sim/test_engine.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add sim/engine.py tests/sim/test_engine.py
git commit -m "feat(sim): simulation engine — compose signal/exit/sizing into trades"
```

---

### Task 8: Validation (splits, walk-forward, Monte-Carlo)

**Files:**
- Create: `sim/validation.py`
- Test: `tests/sim/test_validation.py`

**Interfaces:**
- Consumes: `MarketPath` (`sim.data`); `Trade` (`sim.engine`).
- Produces:
  - `temporal_split(paths, train_frac=0.6, embargo_s=900) -> (train, test)` — sorts by `close_ts`, drops test markets whose `close_ts < max_train_close_ts + embargo_s`.
  - `train_val_test_split(paths, fracs=(0.6, 0.2, 0.2), embargo_s=900) -> (train, val, test)`
  - `walk_forward(paths, n_folds=4) -> list[(train, test)]`
  - `monte_carlo_pvalue(trades, n_iter=1000, seed=42) -> dict` with keys `actual_pnl`, `p_value`, `perm_mean` — each trade wins with probability `implied_prob`; p-value = fraction of permutations whose total P&L ≥ actual.

- [ ] **Step 1: Write the failing test**

```python
# tests/sim/test_validation.py
import pytest
from sim.data import MarketPath, Poll
from sim.engine import Trade
from sim.validation import (temporal_split, train_val_test_split,
                            walk_forward, monte_carlo_pvalue)


def _paths(n):
    return [MarketPath(f"M{i}", 300, 1000 + i * 10_000, i % 2,
                       [Poll(200, 0.5, 0.5, {})]) for i in range(n)]


def test_temporal_split_disjoint_and_ordered():
    train, test = temporal_split(_paths(10), train_frac=0.6, embargo_s=900)
    train_tickers = {p.ticker for p in train}
    test_tickers = {p.ticker for p in test}
    assert train_tickers.isdisjoint(test_tickers)
    assert len(train) == 6
    # embargo: all test closes are >= max train close + 900 (10k spacing => fine)
    max_train = max(p.close_ts for p in train)
    assert all(p.close_ts >= max_train + 900 for p in test)


def test_train_val_test_sizes():
    train, val, test = train_val_test_split(_paths(10), fracs=(0.6, 0.2, 0.2))
    assert (len(train), len(val), len(test)) == (6, 2, 2)


def test_walk_forward_folds():
    folds = walk_forward(_paths(12), n_folds=4)
    assert len(folds) == 4
    for train, test in folds:
        assert {p.ticker for p in train}.isdisjoint({p.ticker for p in test})


def test_monte_carlo_pvalue_strong_edge_is_significant():
    # 20 trades, each implied_prob 0.5 but ALL won with +1 pnl -> highly unlikely
    trades = [Trade(f"M{i}", "no", 1, 0.5, 0.5, 1.0, True, "resolution")
              for i in range(20)]
    res = monte_carlo_pvalue(trades, n_iter=1000, seed=42)
    assert res["actual_pnl"] == pytest.approx(20.0)
    assert res["p_value"] < 0.05
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/sim/test_validation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sim.validation'`

- [ ] **Step 3: Write minimal implementation**

```python
# sim/validation.py
"""Out-of-sample validation: temporal splits, walk-forward, Monte-Carlo test."""
from __future__ import annotations

import random


def _sorted(paths):
    return sorted(paths, key=lambda p: p.close_ts)


def temporal_split(paths, train_frac: float = 0.6, embargo_s: int = 900):
    ordered = _sorted(paths)
    cut = int(len(ordered) * train_frac)
    train, test = ordered[:cut], ordered[cut:]
    if train:
        max_train = max(p.close_ts for p in train)
        test = [p for p in test if p.close_ts >= max_train + embargo_s]
    return train, test


def train_val_test_split(paths, fracs=(0.6, 0.2, 0.2), embargo_s: int = 900):
    ordered = _sorted(paths)
    n = len(ordered)
    c1 = int(n * fracs[0])
    c2 = c1 + int(n * fracs[1])
    train, val, test = ordered[:c1], ordered[c1:c2], ordered[c2:]
    if train:
        mt = max(p.close_ts for p in train)
        val = [p for p in val if p.close_ts >= mt + embargo_s]
    if val:
        mv = max(p.close_ts for p in val)
        test = [p for p in test if p.close_ts >= mv + embargo_s]
    return train, val, test


def walk_forward(paths, n_folds: int = 4):
    ordered = _sorted(paths)
    n = len(ordered)
    fold = n // (n_folds + 1)
    out = []
    for k in range(1, n_folds + 1):
        train = ordered[: fold * k]
        test = ordered[fold * k: fold * (k + 1)]
        if train and test:
            out.append((train, test))
    return out


def monte_carlo_pvalue(trades, n_iter: int = 1000, seed: int = 42) -> dict:
    if not trades:
        return {"actual_pnl": 0.0, "p_value": 1.0, "perm_mean": 0.0}
    actual = sum(t.pnl for t in trades)
    # each trade: win pnl vs loss pnl reconstructed from entry_cost & contracts
    win_pnl = [(1.0 - t.entry_cost_per) * t.contracts for t in trades]
    loss_pnl = [-t.entry_cost_per * t.contracts for t in trades]
    probs = [t.implied_prob for t in trades]
    rng = random.Random(seed)
    ge = 0
    total = 0.0
    for _ in range(n_iter):
        s = sum(win_pnl[i] if rng.random() < probs[i] else loss_pnl[i]
                for i in range(len(trades)))
        total += s
        if s >= actual:
            ge += 1
    return {"actual_pnl": actual, "p_value": ge / n_iter,
            "perm_mean": total / n_iter}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/sim/test_validation.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add sim/validation.py tests/sim/test_validation.py
git commit -m "feat(sim): validation — temporal/train-val-test splits, walk-forward, Monte-Carlo"
```

---

### Task 9: Search + robust objective + gates

**Files:**
- Create: `sim/search.py`
- Test: `tests/sim/test_search.py`

**Interfaces:**
- Consumes: `simulate` (`sim.engine`); `compute_metrics` (`sim.metrics`); `SIGNALS` (`sim.signals`); `EXITS` (`sim.exits`); `SIZERS` (`sim.sizing`); `monte_carlo_pvalue`, `walk_forward` (`sim.validation`); `CostModel` (`sim.costs`).
- Produces:
  - `build_grid(space: dict) -> list[dict]` — Cartesian product of parameter lists.
  - `Gates` dataclass: `min_trades=30`, `max_drawdown=50.0`, `mc_pvalue=0.05`, `min_folds_positive=3`.
  - `objective_score(metrics: dict) -> float` — returns `sharpe` (tiebreak folded in by caller).
  - `evaluate_config(cfg, paths, cost_model) -> (list[Trade], dict)` — runs `simulate` with the signal/exit/sizing named in `cfg["signal"]/["exit"]/["sizing"]` and returns `(trades, metrics)`.
  - `passes_gates(trades, metrics, wf_positive_folds, mc, gates) -> bool`
  - `run_search(space, train, val, test, cost_model, gates, n_folds=4) -> list[dict]` — ranked leaderboard; each row is a dict with `config`, train/val/test metrics, `mc_pvalue`, `wf_positive_folds`, `passed`, `score`.

- [ ] **Step 1: Write the failing test**

```python
# tests/sim/test_search.py
import pytest
from sim.data import MarketPath, Poll
from sim.costs import CostModel
from sim.search import (build_grid, evaluate_config, objective_score,
                        passes_gates, run_search, Gates)


def test_build_grid_cartesian():
    grid = build_grid({"a": [1, 2], "b": [3]})
    assert grid == [{"a": 1, "b": 3}, {"a": 2, "b": 3}]


def _winning_paths(n):
    # bearish NO signals that all resolve NO -> reliable winners
    return [MarketPath(f"M{i}", 300, 1000 + i * 10_000, 0,
                       [Poll(200, 0.40, 0.10, {})]) for i in range(n)]


def _base_config():
    return dict(signal="ensemble", exit="hold", sizing="flat",
                mispricing_threshold=0.25, yes_cutoff=0.72, max_entry_yes=0.65,
                max_entry_no=0.80, no_max_p_raw=0.20, cutoff_buffer=0.03,
                min_entry_price=0.05, min_seconds=60, max_seconds=300,
                mr_return_5m=40.0, mr_price_floor=0.60,
                base_size=1.0, kelly_fraction=0.5, max_size=20.0,
                tp_abs=0.0, sl_abs=0.0, trail_abs=0.0)


def test_evaluate_config_runs():
    trades, metrics = evaluate_config(_base_config(), _winning_paths(40),
                                      CostModel())
    assert metrics["n_trades"] == 40 and metrics["win_rate"] == 1.0


def test_objective_score_is_sharpe():
    assert objective_score({"sharpe": 1.23}) == 1.23


def test_passes_gates_rejects_thin_sample():
    gates = Gates(min_trades=30)
    assert passes_gates([], {"n_trades": 5, "max_drawdown": 0.0}, 4,
                        {"p_value": 0.01}, gates) is False


def test_run_search_ranks_and_gates():
    paths = _winning_paths(60)
    space = {"signal": ["ensemble"], "exit": ["hold"], "sizing": ["flat"],
             "mispricing_threshold": [0.25], "max_entry_no": [0.80]}
    board = run_search(space, paths, paths, paths, CostModel(), Gates(),
                       n_folds=4)
    assert len(board) == 1
    row = board[0]
    assert "config" in row and "score" in row and "passed" in row
    assert row["test_metrics"]["n_trades"] > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/sim/test_search.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sim.search'`

- [ ] **Step 3: Write minimal implementation**

```python
# sim/search.py
"""Grid search with a robust risk-adjusted objective and hard OOS gates."""
from __future__ import annotations

import itertools
from dataclasses import dataclass

from sim.costs import CostModel
from sim.engine import simulate
from sim.metrics import compute_metrics
from sim.signals import SIGNALS
from sim.exits import EXITS
from sim.sizing import SIZERS
from sim.validation import monte_carlo_pvalue, walk_forward


@dataclass
class Gates:
    min_trades: int = 30
    max_drawdown: float = 50.0
    mc_pvalue: float = 0.05
    min_folds_positive: int = 3


def build_grid(space: dict) -> list:
    keys = list(space.keys())
    return [dict(zip(keys, combo)) for combo in itertools.product(*space.values())]


def _defaults() -> dict:
    return dict(signal="ensemble", exit="hold", sizing="flat",
                mispricing_threshold=0.25, yes_cutoff=0.72, max_entry_yes=0.65,
                max_entry_no=0.80, no_max_p_raw=0.20, cutoff_buffer=0.03,
                min_entry_price=0.05, min_seconds=60, max_seconds=300,
                mr_return_5m=40.0, mr_price_floor=0.60,
                base_size=1.0, kelly_fraction=0.5, max_size=20.0,
                tp_abs=0.0, sl_abs=0.0, trail_abs=0.0)


def evaluate_config(cfg: dict, paths, cost_model: CostModel):
    full = _defaults()
    full.update(cfg)
    trades = simulate(paths, SIGNALS[full["signal"]], EXITS[full["exit"]],
                      SIZERS[full["sizing"]], full, cost_model)
    metrics = compute_metrics([t.pnl for t in trades],
                              [t.contracts for t in trades])
    return trades, metrics


def objective_score(metrics: dict) -> float:
    return metrics.get("sharpe", 0.0)


def passes_gates(trades, metrics, wf_positive_folds, mc, gates: Gates) -> bool:
    return (metrics["n_trades"] >= gates.min_trades
            and metrics["max_drawdown"] <= gates.max_drawdown
            and mc["p_value"] < gates.mc_pvalue
            and wf_positive_folds >= gates.min_folds_positive)


def run_search(space, train, val, test, cost_model, gates: Gates,
               n_folds: int = 4) -> list:
    board = []
    for cfg in build_grid(space):
        tr_trades, tr_metrics = evaluate_config(cfg, train, cost_model)
        _, val_metrics = evaluate_config(cfg, val, cost_model)
        test_trades, test_metrics = evaluate_config(cfg, test, cost_model)

        wf_pos = 0
        for f_train, f_test in walk_forward(train, n_folds=n_folds):
            _, fm = evaluate_config(cfg, f_test, cost_model)
            if fm["total_pnl"] > 0:
                wf_pos += 1
        mc = monte_carlo_pvalue(test_trades)
        passed = passes_gates(test_trades, test_metrics, wf_pos, mc, gates)
        board.append({
            "config": cfg,
            "train_metrics": tr_metrics,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "mc_pvalue": mc["p_value"],
            "wf_positive_folds": wf_pos,
            "passed": passed,
            "score": objective_score(test_metrics),
        })
    # rank: passed first, then score, profit-factor tiebreak
    board.sort(key=lambda r: (r["passed"], r["score"],
                              r["test_metrics"]["profit_factor"]), reverse=True)
    return board
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/sim/test_search.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add sim/search.py tests/sim/test_search.py
git commit -m "feat(sim): grid search with robust objective and hard OOS gates"
```

---

### Task 10: Report

**Files:**
- Create: `sim/report.py`
- Test: `tests/sim/test_report.py`

**Interfaces:**
- Consumes: leaderboard rows from `run_search`.
- Produces:
  - `leaderboard_to_dataframe(board: list[dict]) -> pandas.DataFrame` — flattens each row to columns: `signal, exit, sizing, passed, score, test_n_trades, test_win_rate, test_total_pnl, test_profit_factor, test_max_drawdown, mc_pvalue, wf_positive_folds`.
  - `write_report(board: list[dict], out_dir: str) -> dict` — writes `leaderboard.csv` and `summary.md`; returns `{"csv": path, "md": path, "n_passed": int}`. `summary.md` states the top passing config, or explicitly "No strategy passed all gates." when none do.

- [ ] **Step 1: Write the failing test**

```python
# tests/sim/test_report.py
import os
from sim.report import leaderboard_to_dataframe, write_report


def _row(passed, score, pnl):
    return {
        "config": {"signal": "ensemble", "exit": "hold", "sizing": "flat"},
        "train_metrics": {}, "val_metrics": {},
        "test_metrics": {"n_trades": 40, "win_rate": 0.8, "total_pnl": pnl,
                         "profit_factor": 2.0, "max_drawdown": 3.0},
        "mc_pvalue": 0.01, "wf_positive_folds": 4,
        "passed": passed, "score": score,
    }


def test_leaderboard_dataframe_columns():
    df = leaderboard_to_dataframe([_row(True, 1.0, 50.0)])
    assert df.iloc[0]["signal"] == "ensemble"
    assert df.iloc[0]["test_total_pnl"] == 50.0
    assert df.iloc[0]["passed"] == True  # noqa: E712


def test_write_report_with_passing(tmp_path):
    out = write_report([_row(True, 1.0, 50.0)], str(tmp_path))
    assert os.path.exists(out["csv"]) and os.path.exists(out["md"])
    assert out["n_passed"] == 1
    assert "ensemble" in open(out["md"]).read()


def test_write_report_no_passing_states_it(tmp_path):
    out = write_report([_row(False, -1.0, -50.0)], str(tmp_path))
    assert out["n_passed"] == 0
    assert "No strategy passed all gates" in open(out["md"]).read()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/sim/test_report.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sim.report'`

- [ ] **Step 3: Write minimal implementation**

```python
# sim/report.py
"""Render search results into a leaderboard CSV and a markdown summary."""
from __future__ import annotations

import os

import pandas as pd


def leaderboard_to_dataframe(board: list) -> pd.DataFrame:
    rows = []
    for r in board:
        cfg, tm = r["config"], r["test_metrics"]
        rows.append({
            "signal": cfg.get("signal"),
            "exit": cfg.get("exit"),
            "sizing": cfg.get("sizing"),
            "passed": r["passed"],
            "score": r["score"],
            "test_n_trades": tm.get("n_trades"),
            "test_win_rate": tm.get("win_rate"),
            "test_total_pnl": tm.get("total_pnl"),
            "test_profit_factor": tm.get("profit_factor"),
            "test_max_drawdown": tm.get("max_drawdown"),
            "mc_pvalue": r["mc_pvalue"],
            "wf_positive_folds": r["wf_positive_folds"],
        })
    return pd.DataFrame(rows)


def write_report(board: list, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    df = leaderboard_to_dataframe(board)
    csv_path = os.path.join(out_dir, "leaderboard.csv")
    md_path = os.path.join(out_dir, "summary.md")
    df.to_csv(csv_path, index=False)

    passed = [r for r in board if r["passed"]]
    lines = ["# Strategy Search Summary", ""]
    if passed:
        top = passed[0]
        cfg = top["config"]
        tm = top["test_metrics"]
        lines += [
            f"**Configs evaluated:** {len(board)} — **passed all gates:** {len(passed)}",
            "",
            "## Top passing strategy",
            f"- signal=`{cfg.get('signal')}` exit=`{cfg.get('exit')}` "
            f"sizing=`{cfg.get('sizing')}`",
            f"- test P&L: {tm.get('total_pnl'):.2f} | win rate: "
            f"{tm.get('win_rate'):.1%} | profit factor: {tm.get('profit_factor'):.2f}",
            f"- max drawdown: {tm.get('max_drawdown'):.2f} | "
            f"MC p-value: {top['mc_pvalue']:.3f} | "
            f"walk-forward positive folds: {top['wf_positive_folds']}",
        ]
    else:
        lines += [
            f"**Configs evaluated:** {len(board)}",
            "",
            "## No strategy passed all gates",
            "No configuration cleared the out-of-sample, walk-forward, "
            "Monte-Carlo, drawdown, and trade-count gates. This is a valid "
            "negative result: on this data, no robust edge was found.",
        ]
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return {"csv": csv_path, "md": md_path, "n_passed": len(passed)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/sim/test_report.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add sim/report.py tests/sim/test_report.py
git commit -m "feat(sim): report — leaderboard CSV + honest markdown summary"
```

---

### Task 11: Promotion bridge + CLI runner

**Files:**
- Create: `sim/promote.py`
- Create: `sim/run_search.py`
- Test: `tests/sim/test_promote.py`

**Interfaces:**
- Consumes: a leaderboard row `dict` (from `run_search`); `load_paths` (`sim.data`); `train_val_test_split` (`sim.validation`); `run_search`, `Gates` (`sim.search`); `write_report` (`sim.report`); `CostModel` (`sim.costs`).
- Produces:
  - `config_to_settings(cfg: dict) -> dict[str, str]` — maps a config to live `AppSettings` keys: `mispricing_threshold`, `max_entry_price_yes`, `max_entry_price_no`, `no_max_p_raw`, `yes_cutoff` (all stringified). Only keys present in `cfg` are emitted.
  - `promotion_candidate(board: list[dict]) -> dict | None` — returns the top passing row, else None.
  - `sim/run_search.py` `main(argv=None) -> int` CLI: `--data PATH --out DIR`; loads paths, splits, runs a default search space, writes report; prints the promotion candidate's settings or "no candidate".

- [ ] **Step 1: Write the failing test**

```python
# tests/sim/test_promote.py
from sim.promote import config_to_settings, promotion_candidate


def test_config_to_settings_maps_known_keys():
    cfg = {"signal": "ensemble", "mispricing_threshold": 0.2,
           "max_entry_yes": 0.6, "max_entry_no": 0.75, "no_max_p_raw": 0.18,
           "yes_cutoff": 0.7}
    s = config_to_settings(cfg)
    assert s["mispricing_threshold"] == "0.2"
    assert s["max_entry_price_yes"] == "0.6"
    assert s["max_entry_price_no"] == "0.75"
    assert s["no_max_p_raw"] == "0.18"
    assert s["yes_cutoff"] == "0.7"
    assert "signal" not in s  # not a live setting key


def test_config_to_settings_only_present_keys():
    s = config_to_settings({"mispricing_threshold": 0.25})
    assert s == {"mispricing_threshold": "0.25"}


def test_promotion_candidate_prefers_passing():
    board = [
        {"passed": False, "score": 5.0, "config": {"a": 1}},
        {"passed": True, "score": 1.0, "config": {"a": 2}},
    ]
    assert promotion_candidate(board)["config"] == {"a": 2}


def test_promotion_candidate_none_when_no_pass():
    assert promotion_candidate([{"passed": False, "score": 1.0}]) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/sim/test_promote.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sim.promote'`

- [ ] **Step 3: Write minimal implementation**

```python
# sim/promote.py
"""Map a winning search config to live AppSettings keys."""
from __future__ import annotations

# search-config key -> live AppSettings key
_KEY_MAP = {
    "mispricing_threshold": "mispricing_threshold",
    "max_entry_yes": "max_entry_price_yes",
    "max_entry_no": "max_entry_price_no",
    "no_max_p_raw": "no_max_p_raw",
    "yes_cutoff": "yes_cutoff",
}


def config_to_settings(cfg: dict) -> dict:
    out = {}
    for cfg_key, setting_key in _KEY_MAP.items():
        if cfg_key in cfg:
            out[setting_key] = str(cfg[cfg_key])
    return out


def promotion_candidate(board: list):
    passing = [r for r in board if r.get("passed")]
    if not passing:
        return None
    return max(passing, key=lambda r: r["score"])
```

```python
# sim/run_search.py
"""CLI: run the strategy search end-to-end and write a report."""
from __future__ import annotations

import argparse

from sim.costs import CostModel
from sim.data import load_paths
from sim.validation import train_val_test_split
from sim.search import run_search, Gates
from sim.report import write_report
from sim.promote import config_to_settings, promotion_candidate

DEFAULT_SPACE = {
    "signal": ["ensemble", "mean_reversion"],
    "exit": ["hold", "tp_sl", "trailing"],
    "sizing": ["flat", "kelly", "payoff_aware"],
    "mispricing_threshold": [0.15, 0.20, 0.25],
    "max_entry_no": [0.55, 0.65, 0.80],
    "sl_abs": [0.0, 0.10, 0.15],
    "tp_abs": [0.0, 0.20],
    "trail_abs": [0.0, 0.10],
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="live_training_data_20260625.csv")
    parser.add_argument("--out", default="sim_results")
    args = parser.parse_args(argv)

    paths = load_paths(args.data, min_polls=1)
    train, val, test = train_val_test_split(paths)
    board = run_search(DEFAULT_SPACE, train, val, test, CostModel(), Gates())
    report = write_report(board, args.out)
    print(f"Report: {report['md']} | passed: {report['n_passed']}")

    cand = promotion_candidate(board)
    if cand is None:
        print("No promotion candidate — no strategy passed all gates.")
    else:
        print("Promotion candidate settings:", config_to_settings(cand["config"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/sim/test_promote.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Run the full suite and the CLI smoke test**

Run: `pytest tests/sim/ -v`
Expected: PASS (all tests across Tasks 1–11)

Run: `python -m sim.run_search --data live_training_data_deduped_enriched.csv --out sim_results`
Expected: prints a `Report:` line and either a promotion candidate's settings dict or the "No promotion candidate" message; `sim_results/leaderboard.csv` and `sim_results/summary.md` exist.

- [ ] **Step 6: Commit**

```bash
git add sim/promote.py sim/run_search.py tests/sim/test_promote.py
git commit -m "feat(sim): promotion bridge + end-to-end search CLI"
```

---

## Notes for the implementer

- **New-features/model-variants dimension (spec §6):** the signal/exit/sizing registries make new families additive — a new model variant is a new `signal` entry that reads a re-predicted `p_raw` column; add it as its own task following the Task 4 pattern (write failing test → minimal impl → commit). It is intentionally *not* pre-built here to keep the first plan bounded; the framework supports it without change.
- **Fidelity caveat (spec §4):** exit results are conservative estimates because `price_now` is a mark, not a live bid/ask, at ~15s granularity. Do not present exit-strategy P&L as guaranteed.
- **Honest negative results (spec §2, §7):** `write_report` must emit the "No strategy passed all gates" section rather than crown an ungated winner — this is a tested requirement (`test_write_report_no_passing_states_it`).
- **Agent-assisted layer (spec §8, Approach A):** creative strategy generation and adversarial audit of the top-10 are an orchestration concern run *after* this suite exists (feeding new families in as Task-4-style additions and stress-testing leaderboard winners); they are not code modules in `sim/`.

## Known deviations from spec (deliberate, result-neutral)

- **Serial search (spec §5.9 said joblib-parallel).** `run_search` is serial for testability and determinism. Parallelization is a drop-in optimization: wrap the per-config loop body in a function and map it with `joblib.Parallel`. It does not change results, so it is deferred to a follow-up rather than gating the first working suite. If the default space proves too slow, do this first.
- **Reporting is CSV + markdown, not plot tearsheets (spec §5.10).** The leaderboard CSV + `summary.md` satisfy the §12 success criteria. Per-strategy PNG tearsheets (equity curve, drawdown, calibration) are additive presentation — add as a separate `sim/plots.py` task (matplotlib) if you want visuals; not required for a decision.
