# Quality Gates — Phase 3 (Error Handling + Logging) — Design

**Date:** 2026-07-21
**Status:** Approved for planning
**Author:** Claude + Pranaav

Phase 3 of the charter. Phases 1 (lint/type ratchet) and 2 (perf baselines) are merged.

---

## 1. Purpose

Harden error handling and logging via **high-leverage foundation + targeted fixes**:
a global Flask error handler (graceful JSON for all API endpoints), central structured
logging (timestamp/level/name; optional rotating file), fixes for the audit's concrete
silent-failure / missing-traceback / critical sites, and ruff rules that freeze the
broad/bare-except count so new silent catches can't creep in.

## 2. Audit findings (basis for scope)

- Bare `except:`: **0**. Broad `except Exception`: **47** (many legitimate — the scheduler
  wraps every job intentionally). **13 silent swallows** (no log). **9** log without a stack
  trace. Scheduler resilience is already strong.
- **28 of 44 API endpoints have no try/except**; **no global `@app.errorhandler`** → raw 500 HTML.
- **No central logging config** — no format/timestamp; stdout only (gunicorn→Render). No rotation.
- 4 Critical silent-failure sites: `kalshi_trader.get_open_positions` (`return []` on HTTP error, no log); `model_loader._load_from_disk` (unguarded `joblib.load`); `model_loader.predict_proba_raw` (unguarded `predict`); `db_helpers.get_setting` (silent `SQLAlchemyError` retries).

## 3. Goals / Non-goals

### Goals
- Central logging (`configure_logging`) in `create_app`: stdout StreamHandler, format
  `%(asctime)s %(levelname)s [%(name)s] %(message)s`, level from `LOG_LEVEL` (default INFO);
  add a `RotatingFileHandler` **only if `LOG_FILE` env is set** (size cap + backups). Idempotent.
- Global `@app.errorhandler(Exception)`: pass HTTPException through unchanged (preserve 404/405);
  for other exceptions log with stack trace (`logger.exception`) and return graceful output —
  JSON `{"error": "internal error", "status": 500}` for `/api/*`, a minimal HTML message otherwise.
- Fix the audit's concrete list: the **4 Critical** + **13 silent swallows** (add logging) +
  **9 missing-traceback** (→ `logger.exception`/`exc_info=True`), with tests where feasible.
- Enable ruff **E722** (bare except) + **BLE001** (blind except) in `pyproject.toml`; re-seed the
  ruff baseline once (rule addition raises the count legitimately), then it only falls.

### Non-goals
- NOT narrowing the ~47 legitimate broad `except Exception` to specific types (the scheduler's
  job-boundary catches are correct); the global handler + logging fixes cover the real risk.
- NOT adding per-endpoint try/except to all 28 (the global handler covers them in one move).
- No changes to scheduler resilience (already good).
- No coverage push (Phase 4).

## 4. Architecture

### 4.1 Central logging — `app/logging_config.py`
- `configure_logging(app)`: reads `LOG_LEVEL` (default `INFO`) and `LOG_FILE` (optional).
  Attaches a stdout `StreamHandler` with the format above to the root logger (idempotent —
  guard against duplicate handlers on re-init, important because tests build many apps).
  If `LOG_FILE` set, add `RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=3)`.
- Called from `create_app()` early. Does not break the existing `logging.getLogger(__name__)`
  usage in every module (they inherit root config).

### 4.2 Global error handler — in `create_app()`
```python
from werkzeug.exceptions import HTTPException

@app.errorhandler(Exception)
def _handle_unexpected(e):
    if isinstance(e, HTTPException):
        return e  # preserve 404/405/etc.
    app.logger.exception("Unhandled exception on %s", request.path)
    if request.path.startswith("/api/"):
        return jsonify({"error": "internal error", "status": 500}), 500
    return "Something went wrong. Please retry.", 500
```

### 4.3 Targeted fixes (concrete, from the audit)
- **Critical:** `kalshi_trader.get_open_positions` → `logger.warning`/`exception` before `return []`;
  `model_loader._load_from_disk` → wrap `joblib.load` in try/except, log + `return None` (lets the
  DB fallback run); `model_loader.predict_proba_raw` → wrap `predict`, log context, re-raise (do NOT
  fabricate a probability — a bad model must fail loudly, not silently mis-price); `db_helpers.get_setting`
  → `logger.warning` on first retry, `logger.error` on second.
- **13 silent swallows:** add an appropriate `logger.warning`/`logger.exception`.
- **9 missing-traceback:** change `logger.error("...: %s", exc)` → `logger.exception("...")` (or add
  `exc_info=True`). Includes `kalshi_auth`, `kalshi_client:114`, `kalshi_trader:79/254/336`, etc.
- `model_loader._load_from_db` warning → `logger.error`/`exception` (model-load failure isn't a warning).

### 4.4 Regression guard — ruff rules + baseline re-seed
- `pyproject.toml` `[tool.ruff.lint] select` gains `E722`, `BLE001`.
- Enabling BLE001 surfaces the broad-excepts as violations → `ruff_violations` jumps (~+47).
  This is a legitimate rule-addition, so **re-seed** the baseline once via
  `check_quality.py --init` (now perf-preserving from Phase 2). Afterward the higher count is the
  frozen ceiling; it only falls as future narrowing happens.

## 5. Testing

- **Logging:** `configure_logging` attaches a stdout handler with the timestamp format; idempotent
  (calling twice doesn't duplicate handlers); adds a file handler when `LOG_FILE` is set (tmp path).
- **Error handler:** register a temporary route that raises; assert `/api/...` path → 500 JSON
  `{"error","status"}` and the exception is logged; a 404 still returns 404 (HTTPException preserved).
  Use the `client` fixture + a throwaway route or an existing endpoint monkeypatched to raise.
- **Targeted fixes:** where a fix adds a log on a failure path, test that the failure path logs and
  returns the safe value (e.g., `get_open_positions` on a monkeypatched HTTP error logs + returns `[]`;
  `_load_from_disk` on a corrupt path logs + returns `None`).
- **ruff baseline:** after enabling rules + re-seed, `check_quality.py --check-only` passes at the new
  ceiling; E722 stays 0.
- Full suite green throughout; no module-level `from app import` in tests.

## 6. Risks & mitigations

| Risk | Mitigation |
|---|---|
| `errorhandler(Exception)` swallows 404s as 500 | Explicit `isinstance(e, HTTPException): return e` passthrough |
| Duplicate log handlers (tests build many apps) | `configure_logging` is idempotent (checks for existing handler) |
| Enabling BLE001 breaks the gate (+47 violations) | One-time `--init` re-seed of the ruff baseline; documented as rule-addition |
| `predict_proba_raw` fabricated fallback mis-prices trades | Do NOT fabricate — log + re-raise; scheduler's outer catch skips the cycle safely |
| File rotation irrelevant on Render | File handler only when `LOG_FILE` set; stdout is the Render path |

## 7. Success criteria

- `create_app` configures central logging (timestamped stdout; optional rotating file) and registers
  a global error handler that returns graceful JSON for `/api/*` and preserves HTTPExceptions.
- The 4 Critical + 13 silent + 9 tracebackless sites from the audit are fixed and, where practical, tested.
- ruff E722+BLE001 enabled; baseline re-seeded; `--check-only` green at the new ceiling; E722=0.
- Full suite green; the deterministic + perf gates still behave as in Phases 1–2.
