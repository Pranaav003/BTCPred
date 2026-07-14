# BTC-15m Strategy Research & Backtesting Suite — Design

**Date:** 2026-07-14
**Status:** Approved for planning
**Author:** Claude + Pranaav

---

## 1. Purpose

Build a comprehensive, testable strategy-research platform for the Kalshi BTC 15-minute
binary market that can:

- **Backtest** strategies against real logged data with a realistic cost model.
- Report **earning** (P&L, EV, profit factor, drawdown) and **accuracy** (calibration,
  Brier, reliability) metrics.
- **Search** a large strategy space across four dimensions (signal rules, path-dependent
  exits, payoff-aware sizing, new features/models) and crown the best strategy under a
  **robust risk-adjusted objective with hard out-of-sample gates**.
- **Promote** a winning configuration to the live auto-trader, guarded by an OOS gate.

The overriding trade design constraint is **large profits, small losses** — which is why
path-dependent exits (impossible in the current backtester) are a first-class capability.

## 2. Background & motivating findings

- Live net all-time is **−$68.46** at a 58.7% win rate. That win rate is *below* the
  breakeven win rate implied by the payoff geometry (avg win ≈ +$0.30, avg loss ≈ −$0.62 ⇒
  breakeven ≈ 67%). The strategy is structurally underwater even at 1-contract sizing.
- The existing `backtest_v2.py` collapses each market to **one row** and **holds every trade
  to resolution**. This bakes in the binary win/loss payoff and makes "big profits / small
  losses" unreachable by threshold tuning alone.
- The raw per-poll log `live_training_data_20260625.csv` (125,412 rows, ~15s cadence)
  contains each market's **intra-market price trajectory** — enabling exit strategies
  (stop-loss / take-profit / trailing / early exit) for the first time.
- The current strategy's Monte-Carlo permutation p-value is **0.171** — statistically
  indistinguishable from random. Robustness, not raw in-sample P&L, must drive ranking.

## 3. Goals / Non-goals

### Goals
- A modular, unit-tested `sim/` package with an event-driven, path-aware backtester.
- Search across all four dimensions the user approved.
- Robust objective: OOS Sharpe (profit-factor tiebreak) gated on walk-forward + Monte-Carlo
  significance + max-drawdown cap + min-trade-count, reported on an untouched test partition.
- Honest reporting, including the negative result "no strategy passes the gates."
- Promotion path for signal + sizing params into live `AppSettings`, behind the existing
  `validate_retrain.py`-style gate.

### Non-goals (this project)
- **Live execution of exit strategies.** Wiring intra-market sells into the scheduler needs
  new sell-order + monitoring code and is deferred to Phase 2, gated on Phase 1 finding an
  exit strategy that survives the hard gates.
- Replacing the live trading loop, dashboard, or resolver.
- Sourcing new market data feeds (order-book depth, etc.) — we use what is already logged.

## 4. Data sources

| File | Rows | Use |
|---|---|---|
| `live_training_data_20260625.csv` | 125,412 (per-poll ~15s) | **Primary** — rebuild per-market price paths for exits |
| `live_training_data_deduped_enriched.csv` | 15,106 (one/market/bucket) | Hold-to-resolution baseline & signal-rule search (matches `backtest_v2`) |
| `kalshi_btc15m_dataset_scraped.csv` | 11,516 | Model/feature experiments (BTC USD price features) |
| `raw_feature_model.pkl` | — | Current deployed model (CalibratedRandomForest, 33 features) |

**Known data limitations (must be encoded in the design, not hidden):**
- `price_now` = last-trade candle close (a mark), **not** a live bid/ask. Exit fills are
  modeled conservatively (cross spread + fee).
- Poll cadence ~15s ⇒ stops resolve at poll granularity; intrabar spikes not captured ⇒
  model stops conservatively (assume you get the next observed poll price, worst-case within
  the gap for stop tests, with a sensitivity knob).
- `bid_ask_spread` and (in live data) `distance_from_strike` are unpopulated ⇒ not usable as
  live features. `distance_from_strike` exists only in the scraped dataset.

## 5. Architecture

New package `sim/`, each module an isolated, independently-testable unit.

### 5.1 `sim/data.py`
- Loads per-poll CSV, groups by `(market_ticker, entry_bucket)`, orders by descending
  `seconds_to_close`, and builds a `MarketPath` dataclass:
  `ticker, bucket, close_ts, final_outcome_yes, polls: list[Poll]` where each `Poll` carries
  `seconds_to_close, price_now, p_raw, features(dict)`.
- Caches parsed paths to parquet/pickle for fast repeated runs.
- Loader for the scraped dataset for model/feature work.
- **Interface:** `load_paths(path, min_polls=…) -> list[MarketPath]`.

### 5.2 `sim/costs.py` (single source of truth for costs)
- Fixes today's inconsistency (`backtest_v2` uses $0.02 spread + 1% fee; analysis scripts use
  aggressive offsets). All modules import from here.
- `entry_cost(side, price_now, cfg) -> per_contract_cost`
- `exit_cost(side, price_now, cfg) -> per_contract_proceeds` (selling crosses the spread the
  other way; fee applied on gains).
- `kalshi_fee(gross) -> fee`.
- Config object `CostModel` with tunable spread, fee rate, and slippage offsets.

### 5.3 `sim/signals.py`
- Pluggable entry-signal callables: `signal(path, params) -> EntryDecision | None`
  (`side`, entry poll index).
- Includes a parameterized port of `evaluate_ensemble_signal` plus new families (§6).

### 5.4 `sim/exits.py`
- Pluggable exit policies: `exit_policy(path, entry_idx, side, params) -> (exit_idx, exit_price, reason)`.
- Policies: hold-to-resolution (baseline), take-profit, stop-loss, trailing stop,
  breakeven-lock-after-+X, time-decay exit, volatility-scaled stop, and combinations.

### 5.5 `sim/sizing.py`
- Pluggable sizing: `size(edge, payoff_ratio, bankroll, params) -> contracts`.
- Modes: flat, fractional-Kelly, payoff-ratio-aware (skip trades that can't clear their
  breakeven win-rate), edge-scaled, drawdown-capped.

### 5.6 `sim/engine.py`
- `simulate(paths, signal, exit_policy, sizing, params) -> TradeLedger`.
- One trade per market at earliest qualifying signal (no look-ahead; matches existing
  convention). Deterministic. Mark-to-market walk of the path for exit evaluation.

### 5.7 `sim/metrics.py`
- **Earning:** total P&L, EV/contract, avg win/loss, profit factor, Sharpe, Sortino, max
  drawdown, exposure, turnover.
- **Accuracy:** Brier, log-loss, reliability curve, realized win-rate vs breakeven win-rate.
- All metrics available split by YES/NO and by entry-time bucket.

### 5.8 `sim/validation.py`
- Market-level temporal **walk-forward** CV with 15-minute embargo (reuse `temporal_split`
  semantics).
- **Monte-Carlo permutation** test (seeded, deterministic).
- **train → validation → test** protocol: search & rank on train, gate on validation, report
  the honest number on an untouched test partition. Guards against selection bias from
  thousands of trials.

### 5.9 `sim/search.py`
- Parallel grid + random/Bayesian search over the combined space (joblib/multiprocessing).
- Ranks by the robust objective (§7). Emits a leaderboard CSV.

### 5.10 `sim/report.py`
- Ranked leaderboard + per-strategy tearsheets (equity curve, drawdown, calibration plot,
  win/loss histograms, YES/NO split) as CSV + markdown (+ optional PNG plots).

### 5.11 `sim/promote.py`
- Maps a winning config → live `AppSettings` keys (`mispricing_threshold`, cutoffs, entry
  caps, sizing; exit params where a Phase-2 executor exists).
- Guarded by a `validate_retrain.py`-style OOS gate; refuses to promote a config that fails.

### 5.12 `tests/sim/`
- pytest per unit: cost math; exit triggers on synthetic hand-built paths; sizing formulas;
  **engine no-look-ahead invariant**; split integrity (no market in both partitions, embargo
  honored); seeded Monte-Carlo determinism; metric formula correctness.

## 6. Creative strategy families

- **Exits:** TP @ +X, SL @ −Y, trailing stop, lock-breakeven-after-+X, time-decay exit,
  volatility-scaled stops.
- **Entries:** fade extreme 15m moves (mean-reversion), momentum continuation, RSI/flip-count
  regime filters, session-specific rules (Asia/EU/US), distance-from-strike bands,
  calibrated-zone-only (NO `p_raw < 0.20`).
- **Sizing:** size ∝ edge × payoff-ratio; refuse trades whose best achievable payoff can't
  beat their breakeven win-rate.
- **Models:** gradient boosting / logistic / stacked ensembles; isotonic-vs-sigmoid
  recalibration; per-regime models — trained on the 33 already-logged features.

## 7. Objective function

Rank by **OOS Sharpe** (profit-factor tiebreak), among configs passing **all** hard gates:
- Walk-forward: net-positive in **≥3 of 4 folds** AND positive on the final holdout fold.
- Monte-Carlo permutation **p < 0.05** on the holdout.
- **Max drawdown ≤ cap** (configurable; default expressed as % of deployed bankroll).
- **≥ 30 trades** (no thin samples).
- Reported performance comes from the untouched **test** partition only.

If no config passes, the suite reports that explicitly rather than crowning an overfit
winner.

## 8. Recommended approach

**Approach A:** unified event-driven path-backtester + parallel Python search +
agent-assisted creative generation and adversarial validation. Python does the heavy,
deterministic, testable numeric sweep; LLM workflow agents (a) generate novel strategy
families and (b) adversarially audit the top-10 winners for overfitting, leakage, and
regime-dependence before any are trusted or promoted.

Rejected: pure-Python-no-agents (narrower creativity, no adversarial review); extending
`backtest_v2.py` in place (structurally cannot do exits).

## 9. Phasing

- **Phase 1 (this project):** full research suite; promote signal + sizing params to live;
  exits proven in simulation only.
- **Phase 2 (deferred):** live execution of winning exit strategies (new sell-order +
  monitoring code), gated on Phase 1 producing an exit strategy that survives the hard gates.

## 10. Proposed file layout

```
sim/
  __init__.py
  data.py
  costs.py
  signals.py
  exits.py
  sizing.py
  engine.py
  metrics.py
  validation.py
  search.py
  report.py
  promote.py
  run_search.py        # CLI entry point
tests/sim/
  test_costs.py
  test_exits.py
  test_sizing.py
  test_engine_no_lookahead.py
  test_metrics.py
  test_validation.py
docs/superpowers/specs/2026-07-14-strategy-search-suite-design.md
```

## 11. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Exit fills over-optimistic (mark ≠ bid/ask) | Conservative cost model + slippage sensitivity sweep |
| Overfitting across thousands of configs | train→validation→test protocol; MC significance; walk-forward |
| No genuine edge exists | Suite reports negative result honestly; cheap to learn |
| Poll-granularity stops | Worst-case-within-gap stop modeling + sensitivity knob |
| Promotion breaks live bot | Reuse `validate_retrain` gate; Phase-2 execution deferred |

## 12. Success criteria

- All `tests/sim/` pass.
- Search runs end-to-end and produces a ranked leaderboard + tearsheets.
- Either a config that passes every hard gate (with honest test-partition numbers) **or** a
  clear, evidenced "no strategy passes" report.
- Winning signal/sizing config expressible as live `AppSettings` and validated by the OOS
  gate.
