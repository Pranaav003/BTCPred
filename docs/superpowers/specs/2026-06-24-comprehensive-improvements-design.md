# BTCPred Comprehensive Improvements Design

**Date:** 2026-06-24
**Status:** Draft
**Scope:** Signal accuracy, live trading reliability, code quality, dashboard UX

## Context

BTCPred is a Kalshi BTC 15-minute market signal dashboard deployed on Render (starter plan). It uses an ensemble of Kalshi market probability and an independent XGBoost/RandomForest model to generate BUY YES/BUY NO signals. Live trading is active with real money on Kalshi.

**Current state:** Live P&L is -$7.42 on a $92.57 balance. Marginal agreement signals (70-71% model confidence) are losing. Multiple IOC orders are going unfilled. The codebase has zero test coverage, unpinned dependencies, and several correctness bugs in financial calculations.

## Approach: Layered Stabilize-then-Enhance

Ship in dependency order: fix what is broken, then harden what is fragile, then enhance what is working.

---

## Phase 1 — Stabilize (Correctness, Safety, Data Integrity)

### 1.1 Critical Bug Fixes

| # | Bug | File | Fix |
|---|---|---|---|
| 1 | PnL miscalculation in `_apply_kalshi_settlement` | `app/resolver.py` | Use `side_cost` (already computed but unused) for PnL: `revenue/100 - side_cost - fee_cost` instead of `revenue/100 - total_cost - fee_cost` |
| 2 | Inconsistent `max_daily_loss` defaults (200.0 vs 50.0) | `app/scheduler.py` | Single constant `DEFAULT_MAX_DAILY_LOSS = 50.0` used by both `_auto_trade_allowed_by_daily_loss` and `_execute_live_trade` |
| 3 | Data leakage in train/test split — same market in both sets | `train_raw_model.py`, `model_comparison.py`, `merge_and_retrain.py` | Split by `market_ticker` not by row index; add 15-minute embargo period between train and test |
| 4 | `seconds_to_close` truncation — `int(89.7)` = 89, bypasses 90s minimum | `app/scheduler.py:378` | Use `math.ceil()` instead of `int()` |
| 5 | CSS syntax error — orphaned closing brace after `.monitor-live-strip` | `app/static/css/main.css:~1700` | Remove the orphaned block |
| 6 | Duplicate features — `momentum_1m/3m/5m` identical to `return_1m/3m/5m`; `price_velocity_5m` is linear rescaling of `return_5m` | `app/feature_engineering.py`, `train_raw_model.py` | Remove duplicates from `RAW_FEATURES`; keep `momentum_acceleration` (genuinely different) |

### 1.2 Safety Guards

- **API input validation:** `/api/paper/trade` must validate `dollar_amount` against portfolio cash on the server side (currently only frontend checks)
- **CSRF protection:** Add `Flask-WTF` CSRF tokens to all POST endpoints
- **Partial fill handling:** In `kalshi_trader.py`, when `1 <= fill_count < count`, record `fill_count` contracts and log a warning instead of treating it identically to a full fill
- **Model artifact:** Move `raw_feature_model.pkl` out of git (add `*.pkl` to `.gitignore`), download at deploy time from Render disk or S3
- **Pin dependencies:** Pin all versions in `requirements.txt` (e.g., `flask==3.0.0`, `scikit-learn==1.5.0`)

### 1.3 Database & Migrations

- Add `Flask-Migrate` (Alembic) for schema migrations
- Remove the SQLite-only `_ensure_signal_schema_columns` hack; let Alembic handle all schema changes
- The current `db.create_all()` does not add columns to existing tables, which means production deployments silently run with missing columns after model changes

---

## Phase 2 — Harden (Reliability, Resilience, Observability)

### 2.1 Test Suite

Add `pytest` with targeted tests for financial correctness (most critical paths):

- **PnL calculation tests:** `compute_pnl()` for all combinations of side/outcome/price
- **Settlement reconciliation tests:** `_apply_kalshi_settlement()` for correct/wrong/unfilled
- **Position sizing tests:** `compute_position_size()` edge cases (0 cash, 100% cash, volatility override)
- **Signal evaluation tests:** `evaluate_signal()` / `evaluate_mispricing_signal()` / `evaluate_ensemble_signal()` with known inputs and expected outputs
- **Feature engineering tests:** `compute_features()` with synthetic candle/trade data
- **API endpoint tests:** basic request/response validation for all POST endpoints

### 2.2 Reliability Improvements

| Area | Current Problem | Fix |
|---|---|---|
| Network retries | `_get()` only retries on 429; connection errors return `None` immediately | Add 1 retry with 2s backoff on `ConnectionError`/`Timeout` |
| BTC price cache lock | `_btc_price_cache` read/written without `_cache_lock` | Use existing `_cache_lock` pattern |
| Kalshi auth key caching | Private key re-parsed from env var on every request | Cache the deserialized key object |
| Cooldown backoff | 3 failures -> 60s cooldown -> reset to 0 failures -> repeat cycle | Exponential backoff: 60s -> 120s -> 240s, cap at 5min |
| Scheduler worker guard | No assertion preventing multi-worker gunicorn from running duplicate schedulers | Add DB-based advisory lock or assert `GUNICORN_WORKERS == 1` |
| Missing features -> 0.0 | `model_loader.py` silently fills missing features with 0.0 (indistinguishable from "flat price") | Log a warning; add missingness indicator columns to model pipeline |
| Free-tier DB expiry | Render free PostgreSQL expires after 90 days with no backup | Add `flask db dump` CLI command; document upgrade path in README |

### 2.3 Observability

- **Model staleness alert:** If `trained_at` is > 7 days old, flash a warning on the dashboard
- **Rolling accuracy tracker:** Track last-50-signal accuracy in DB, surface in `/api/metrics`
- **Health check endpoint:** Wire `/api/health` into `render.yaml`'s `healthCheckPath`
- **Fill rate tracking:** Log IOC fill rate (fills attempted vs. filled) to identify when limit prices are too far from market

---

## Phase 3 — Enhance (Signal Accuracy, UX, Code Quality)

### 3.1 ML Pipeline Overhaul

#### 3.1.1 Fix Data Foundation

- **Market-level train/test split:** Split by `market_ticker` so no market appears in both sets. Add 15-minute embargo between train and test periods. This alone may change which strategies look profitable.
- **Walk-forward validation:** Implement expanding-window cross-validation: train on months 1-N, test on month N+1, expand, repeat. This is the gold standard for financial time-series.
- **Missingness indicators:** Before median imputation, add `was_missing_X` boolean columns for features that are NaN. Let the model learn from the pattern of missing data.
- **Remove duplicate features:** Drop `momentum_1m/3m/5m` (identical to `return_1m/3m/5m`) and `price_velocity_5m` (linear rescaling of `return_5m`). Keep `momentum_acceleration` (genuinely different: `return_1m - return_3m/3`).

#### 3.1.2 New Features

| Feature | Why It Matters |
|---|---|
| `bid_ask_spread` | Wide spread -> execution risk, lower effective edge |
| `volume_acceleration` (`volume_1m / volume_5m`) | Rising trading activity predicts direction changes |
| `trade_intensity` (`trade_count_1m / 1` per minute) | Normalized activity measure |
| `rsi_14` | Standard momentum oscillator, captures overbought/oversold |
| `session` (Asia=0, EU=1, US=2, overlap=3) | BTC has strong intraday patterns by trading session |
| `distance_from_strike` | Model predicts "BTC > strike at expiry" but strike is not a feature |
| `outcome_rate_bucket` | Historical win rate for this entry_bucket as a prior |
| `return_5m_ratio` (`return_1m / return_5m` when return_5m != 0) | Momentum acceleration vs. trend direction |

**Not included (YAGNI):** Orderbook depth (data not available), LSTM/sequential features (architecture change), cross-market features (only one active market).

#### 3.1.3 Model Improvements

- **Probability calibration:** Wrap the RF/XGBoost pipeline with `CalibratedClassifierCV` (isotonic regression). Raw RF `predict_proba` is poorly calibrated. Calibration directly improves signal accuracy because the mispricing threshold and agreement cutoff both depend on `p_raw` being well-calibrated.
- **Stacking ensemble:** Train RF, XGBoost, and LogisticRegression as base models, then a LogisticRegression meta-learner on their `predict_proba` outputs. This typically outperforms any single model on tabular data.
- **Class weighting:** Add `class_weight='balanced'` to RF and LR, `scale_pos_weight` to XGBoost. If YES/NO ratio is imbalanced, balanced weighting prevents majority-class bias.
- **Hyperparameter tuning:** Use `TimeSeriesSplit` cross-validation with Optuna to tune key hyperparameters for each model family.

#### 3.1.4 Backtest Improvements

- **Transaction costs:** Model Kalshi's 1% fee on profits + bid-ask spread (default 2 cents). Currently the backtest assumes free execution at `p_market`.
- **Walk-forward backtest:** Run the backtest in expanding windows instead of a single train/test split.
- **Regime analysis:** Break down results by volatility regime (high/low using `volatility_5m` median split).
- **Proper Sharpe ratio:** Compute from daily PnL returns, not per-trade PnL, and annualize.

#### 3.1.5 Retraining Pipeline

- **Concept drift detection:** Before retraining, run a KS test on key feature distributions (live vs. training). Only retrain if drift is detected.
- **Model comparison guard:** After retraining, compare new model vs. old on the last N resolved signals. Only deploy if Brier score is equal or better. Otherwise, log a warning and keep the old model.
- **Hot model reload:** Add a `/api/model/reload` endpoint that invalidates the `model_loader` cache, so retrained models take effect without a full Flask restart.

### 3.2 Live Trading Strategy Improvements

#### 3.2.1 Settings Changes

| Setting | Current Value | Recommended | Rationale |
|---|---|---|---|
| Aggressive `yes_cutoff` | 0.70 (customized) | 0.75 | Marginal agreement trades at 70-71% are losing. 0.75 creates a 5% buffer. |
| Aggressive `max_seconds` | 480s | 360s | Losing trades entered at 400-470s. 360s = 6 min max, still generous. |
| Max YES entry price | 0.80 | 0.75 | The 0.79 entry was a loser. At 0.75, 25c+ upside per contract. |
| Mispricing threshold | 0.25 | 0.30 | 27.9% bearish gap bypassed volatility guard. 0.30 = fewer but higher-conviction trades. |
| Max reversal risk (mispricing cap) | 0.65 (hardcoded) | 0.50 | Blocks mispricing in high-vol regimes where gaps are likely noise. |

#### 3.2.2 Strategy Logic Changes

| Improvement | Rationale |
|---|---|
| Raise default agreement cutoff from 0.65 -> 0.70 | The 0.65 cutoff produces marginal signals. Higher cutoff filters weak signals. |
| Disable volatility guard bypass for mispricing | In high-vol regimes, mispricing gaps are more likely noise. The bypass was designed for moderate-vol relative-value trades, not extreme conditions. |
| Minimum confidence threshold for auto-trades | Require `confidence >= 0.35` before auto-trading. Currently any actionable signal auto-trades. |
| Adaptive IOC pricing | Track fill rate over last 20 orders. If < 50% fill, reduce spread cushion from +2c to +1c. If > 90% fill, increase from +2c to +3c. Persist fill rate in AppSettings. |
| No-trade zone near cutoff | Add a configurable `cutoff_buffer` setting (default 0.05). Don't trade if `p_raw` is within `cutoff_buffer` of the yes_cutoff. These are the weakest signals. |

#### 3.2.3 Code-Level Fixes

1. **`MAX_MISPRICING_OVERRIDE_RISK`** (`signal_engine.py:14`): Change from 0.65 to 0.50. Currently hardcoded. Make it a DB setting (`max_mispricing_override_risk`) for runtime configurability.
2. **Poll interval wiring** (`scheduler.py:517`): The scheduler ignores the DB `poll_interval` setting and hardcodes 30s. Either read `AppSettings.get("poll_interval_seconds")` or remove the UI control to avoid confusion.

### 3.3 Dashboard UX Improvements

| Area | Current | Improvement |
|---|---|---|
| Expected value display | Shows raw payout | Add EV = `p_correct * payout - (1-p_correct) * cost` alongside raw payout |
| Trade confirmation | BUY YES has no confirmation step | Add confirmation modal for all trades, showing EV and risk |
| Bid/ask spread | Not shown | Display spread as a pill next to prices; warn if spread > 5 cents |
| Drawdown metrics | Only cumulative PnL | Add max drawdown, current drawdown from peak, Sharpe ratio |
| Market closed state | Shows "--" everywhere | Add explicit "Market closed" state with next market countdown |
| Loading states | "--" placeholders | Add skeleton screens for initial load |
| Mobile bottom nav | No safe area insets | Add `env(safe-area-inset-bottom)` for phones with home indicators |
| Chart tooltips | Not touch-friendly | Add Chart.js touch interaction plugin |

### 3.4 Code Quality

| Item | Current | Improvement |
|---|---|---|
| CSS | Single 2960-line `main.css` | Split into `base.css`, `dashboard.css`, `monitor.css`, `analytics.css`, `settings.css` |
| JS | 4 files, `main.js` at 1846 lines | Split `main.js` into `chart.js`, `trading.js`, `polling.js`, `state.js` modules |
| Settings auto-save | `window.confirm()` for discard | Use app's custom modal system |
| Risk profile editing | Re-renders entire grid on every slider input, losing focus | Update only the changed card in-place |
| Analytics charts | Destroy and recreate every 60s (causes flicker) | Update chart data in-place; only recreate if structure changes |

---

## Implementation Priority

Within each phase, items are ordered by impact and dependency:

**Phase 1 (Stabilize):**
1. Bug #1 (PnL miscalculation) — directly affects financial accuracy
2. Bug #3 (data leakage) — affects all ML work
3. Bug #2 (daily loss defaults) — safety
4. Bug #4 (seconds truncation) — safety
5. Bug #5 (CSS syntax error) — quick fix
6. Bug #6 (duplicate features) — prerequisite for ML work
7. Safety guards (API validation, CSRF, partial fills, model artifact, pin deps)
8. Flask-Migrate integration

**Phase 2 (Harden):**
1. Test suite (PnL, settlement, position sizing first)
2. Reliability improvements (network retries, key caching, cooldown backoff)
3. Observability (model staleness, rolling accuracy, health check, fill rate)

**Phase 3 (Enhance):**
1. ML data foundation (market-level split, walk-forward, missingness indicators)
2. New features (bid_ask_spread, volume_acceleration, session, etc.)
3. Model improvements (calibration, stacking, class weighting, tuning)
4. Backtest improvements (transaction costs, walk-forward, regime analysis)
5. Strategy settings and logic changes
6. Retraining pipeline improvements
7. Dashboard UX improvements
8. Code quality (CSS/JS split, settings UX fixes)

---

## Out of Scope

The following were considered but excluded:

- **Full frontend framework migration** (React/Vue) — YAGNI; the Jinja2 + vanilla JS stack works
- **Orderbook depth features** — Kalshi public API does not provide this data
- **LSTM/sequential models** — Would require architecture change from point-in-time features to time-windowed sequences; significant complexity for uncertain gain
- **Cross-market features** — Only one KXBTC15M market is active at a time
- **User authentication** — Single-user app; not needed now
- **CI/CD pipeline** — Important but orthogonal to this design; separate project
- **Database upgrade from Render free tier** — Infrastructure decision separate from code improvements
