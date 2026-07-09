# BTCPred Live-Loss Fix — Design Spec

**Date:** 2026-07-09
**Branch:** `fix/live-loss-retrain-and-guards`
**Author:** RCA + design session (see memory `live-loss-rca-2026-07-09`)

## Problem

The live Kalshi BTC-15m bot is down **−$68.22** (106 filled, 57.6% win rate) even though the
exact rule it runs backtests as *positive*-EV (`backtest_current_rule.py`: +0.082/contract, 66% WR).
Adversarially-verified RCA (2026-07-09) identified four causes, ranked by evidence:

1. **Oversized legacy sizing (dominant, proven).** −$68.22 is mathematically impossible at 1
   contract/trade (max loss = 45 × $0.99 = $44.55). Implied historical size ~12–19 contracts
   (~$7–11/trade). P&L is ~16σ below backtest expectation → structural, not variance. Most damage
   predates the 2026-07-07 `$4` cap + `kelly_lite_v2` sizing and the rate-limit/resting-order bug era.
2. **Threshold config drift.** Live fires on gaps 10–19% ⇒ production `mispricing_threshold ≈ 0.10`,
   not the validated 0.25 (`apply_backtest_settings.py` is a manual one-shot never effective on prod).
   At 0.10 the NO edge is +0.058 with only **5.8¢** slippage cushion; at 0.20 it is +0.202 with 20¢.
3. **Model miscalibration for NO.** The model's real NO edge exists only where `p_raw < 0.20`
   (predicts YES 13.8%, actual 13.4%, 86.6% WR — 85% of aggregate edge). In `p_raw` 0.35–0.50 it
   says YES ≈ 43% but reality is 56.6% and the market (59.4%) is closer. Globally the market is 2×
   better calibrated (MAE 0.015 vs 0.030).
4. **Edge decay / overfit to May.** Backtest EV decays May +0.102 → June +0.013; late-June trades
   negative. Model trained through ~2026-06-25; July not evaluable from the historical CSV.

**Ruled out:** pure variance (16σ); ask-crossing slippage as a *proven* driver (5 live fills show ~0¢
net slippage — but the 5.8¢ cushion at threshold 0.10 makes it a latent risk).

## Goal & decisions

- **Goal:** Restore profitability by *retraining* the stale model, then re-enabling full sizing —
  not merely stopping the bleeding. (User decision.)
- **Interim posture during rebuild:** stay live but **tiny fixed size** (guards + hard 1-contract
  cap) so residual mistakes cost cents while real fills keep logging. (User decision.)
- **Validation gate before restoring full sizing:** *both* (a) new-model Brier < old-model Brier on a
  held-out set, and (b) current-rule replay EV/contract > 0 on a held-out July slice. (User decision —
  lighter gate.)

## Architecture — three sequenced phases

Phase 0 ships immediately (config + code guards). Phase 1 (retrain + validate) depends on a
production data export only the user can trigger. Phase 2 (restore full sizing) is gated on Phase 1
passing.

### Phase 0 — Interim safety (config + code guards)

**Config (production DB — run on Render shell, verify with `diag.py`):**
Apply the validated settings via `apply_backtest_settings.py` (extended to also set the two new keys):
- `mispricing_threshold`: → **0.2500**
- `max_entry_price_yes`: → 0.6500 (NO cap unchanged at 0.80)
- moderate-profile window / cutoffs per existing validated set; `signal_mode` = `ensemble`
- `no_max_p_raw`: **0.20** (new)
- `live_max_contracts`: **1** (new, interim)

**Code changes:**

| File | Change |
|------|--------|
| `app/signal_engine.py` | Add `no_max_p_raw` gate: block `PAPER BUY NO` unless `p_raw < no_max_p_raw`, in `evaluate_ensemble_signal` and `evaluate_mispricing_signal` (new param, default 0.20). Read the setting in `evaluate_live_signal` and thread it through both callsites. Emit a clear NO-SIGNAL reason when blocked. |
| `app/scheduler.py` | In `_execute_live_trade`, after `contracts = int(trade_size / entry_price)`, clamp `contracts = min(contracts, live_max_contracts)` when the setting is a positive int. Recompute `actual_cost`. Existing `$4` cap and `max_risk` unchanged. |
| `app/db_helpers.py` | Add defaults: `no_max_p_raw = "0.20"`, `live_max_contracts = ""` (empty = no cap). |
| `apply_backtest_settings.py` | Add `mispricing_threshold=0.2500`, `no_max_p_raw=0.20`, `live_max_contracts=1` to `VALIDATED_SETTINGS`. |

**Design choices:** both new guards are **tunable AppSettings** (not hardcoded constants), matching
the codebase's settings-driven pattern. `live_max_contracts` empty string ⇒ no cap (Phase 2 clears it).

**Tests (TDD — written before implementation):**
- `tests/test_no_max_p_raw_gate.py`: NO blocked at `p_raw ≥ 0.20`, allowed below; YES path unaffected;
  gate respects the setting value.
- `tests/test_live_max_contracts.py`: contracts clamped to the cap; no clamp when unset; `actual_cost`
  recomputed after clamp.

### Phase 1 — Retrain & validate

1. **Export (user):** `GET /api/export/live-training-data` → save as `live_training_data.csv` (all data
   through July).
2. **Retrain (local):** `python3 merge_and_retrain.py --live-data live_training_data.csv` — existing
   pipeline does market-level temporal split, 900s embargo, KS drift check, 2× live upweight, RandomForest,
   and an old-vs-new Brier comparison. Produces `raw_feature_model.pkl`.
3. **Validate (new `validate_retrain.py`):** the gate. Use the **same market-level temporal test split**
   `merge_and_retrain.py` produces (the most-recent ~20% of markets by `close_ts`, i.e. the July-most
   data the model never trained on). Crucially, **re-predict `p_raw` with the new model on each held-out
   row's raw features** — do NOT reuse the `p_raw` column already in the CSV (that was logged by the old
   model at collection time). Then compute:
   - new-model Brier vs old-model Brier on the held-out set (must be `new < old`);
   - current-rule replay (reuse `backtest_current_rule.py` logic, THRESH=0.25 to match production) EV/contract
     using the new model's re-predicted `p_raw` (must be `> 0`, reported with WR / n / 95% CI).
   Prints `PASS`/`FAIL` with numbers. FAIL ⇒ iterate (live-weight, feature review), do not deploy.
4. **Deploy (on PASS):** upload via `POST /api/model/upload`, hot-reload via `POST /api/model/reload`.
   Keep the previous `.pkl` for rollback.

### Phase 2 — Restore full sizing

After Phase 1 gate passes and the new model is live: clear `live_max_contracts` (remove the 1-contract
cap) to restore edge-scaled sizing. Keep the `$4` hard cap, `max_risk = balance×0.10`, and `max_daily_loss`
as backstops. Monitor via `/api/live/stats`.

## Data flow

```
prod DB signals ──export──▶ live_training_data.csv ──merge_and_retrain──▶ raw_feature_model.pkl
      ▲                                                         │
      │                                          validate_retrain.py (gate)
      │                                                         │ PASS
   scheduler ◀── hot reload ◀── ModelArtifact (prod) ◀── /api/model/upload
```

## Error handling & rollback

- Model deploy is reversible: re-upload the retained previous `.pkl` via `/api/model/upload`.
- `merge_and_retrain.py` already warns if the new Brier is worse; `validate_retrain.py` hard-blocks deploy.
- `max_daily_loss` and the `$4` cap remain active in every phase.
- The `no_max_p_raw` gate degrades safely: if the setting is missing, default 0.20 applies.

## Testing strategy

- Phase 0 logic covered by two new unit-test files (TDD).
- Existing suite (`tests/`) must stay green — especially `test_signal_engine.py`,
  `test_aggressive_entry_price.py`.
- Phase 1 gate (`validate_retrain.py`) is itself the empirical test for model quality; it is run manually
  with the exported data, so it is not part of the unit suite.

## Out of scope

- Dashboard `entry_price` display inconsistency for NO trades (cosmetic; noted separately).
- Feature-engineering changes / new model architecture (retrain reuses the current RandomForest pipeline).
- Any change to the paper-trading path beyond what the shared signal gate implies.

## Rollout order

1. Phase 0 code + tests (this branch) → merge → deploy → run `apply_backtest_settings.py` on Render →
   verify with `diag.py` (confirm `mispricing_threshold=0.25`, `live_max_contracts=1`, `no_max_p_raw=0.20`).
2. Phase 1 export → retrain → validate. Iterate until gate passes.
3. Phase 2: clear `live_max_contracts`; monitor.
