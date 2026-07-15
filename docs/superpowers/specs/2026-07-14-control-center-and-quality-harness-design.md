# Strategy Control Center + Quality Harness — Design

**Date:** 2026-07-14
**Status:** Approved for planning
**Author:** Claude + Pranaav

---

## 1. Purpose

Two coordinated deliverables that make BTCPred a "complete project":

1. **Quality harness** — real pytest configuration, shared fixtures, a true end-to-end
   pipeline test, page/API integration tests, a **coverage+test-count baseline that
   ratchets up**, and an **auto-run hook** so tests run on every change and errors are
   caught immediately.
2. **Strategy Control Center** — one clean, task-oriented dashboard page for the modern
   (validated ensemble) strategy that makes the key levers easy to change, defaults to
   **paper** (never live), and shows at a glance whether the bot is actually trading.

The harness is built **first** so the Control Center is developed test-driven and the
ratchet/auto-run is already enforcing quality during the UI build.

## 2. Background

- Current dashboard: Flask + Jinja + a monolithic 1,956-line `main.js` + custom CSS, no
  build step, no auth. The Settings page is a wall of controls that is hard to use/change.
- Current tests: ~157 test functions run via **bare `pytest`** — no `pytest.ini`, no
  `conftest.py`, no coverage, no CI, no hooks, no baseline, **no end-to-end pipeline test**.
- `create_app("testing")` currently `KeyError`s — there is no `TestingConfig`.
- Render DB is **ephemeral** (wiped on each deploy), so seed defaults determine post-deploy
  behavior — today it comes up idle/not paper-trading.

## 3. Goals / Non-goals

### Goals
- Deterministic, configured test runs with coverage.
- A real end-to-end pipeline test (data → features → model → signal → paper trade → resolution).
- A ratcheting baseline that fails on any regression and rises on success.
- A Stop hook that runs the baseline check every turn.
- A focused Control Center page: paper-default, one-click validated config, clear status.
- Fresh deploys come up **paper-trading the validated strategy** (never live).

### Non-goals
- No authentication system (dashboard remains open; noted as a known risk).
- No frontend framework / build step (keep server-rendered Jinja + vanilla JS + CSS).
- No rebuild of Monitor / Analytics / existing Dashboard pages.
- No live-exit execution (the strategy search found exits don't help).
- No refactor of the existing `main.js` (Control Center gets its own focused JS).

## 4. Architecture

### Piece A — Quality Harness

| Component | File | Responsibility |
|---|---|---|
| Pytest + coverage config | `pyproject.toml` | testpaths=`tests`, markers (`unit`/`integration`/`e2e`), `[tool.coverage.run]` source=`app`,`sim` |
| Dev deps | `requirements-dev.txt` | pin `pytest`, `pytest-cov`, `coverage` |
| Shared fixtures | `tests/conftest.py` | `app` (TestingConfig, in-memory SQLite, scheduler off), `client`, tiny model/dataset fixtures; dedupes the `sys.modules` stub pattern |
| Testing config | `app/config.py`, `app/__init__.py` | add `TestingConfig` (TESTING=True, `sqlite:///:memory:`, scheduler suppressed) and register `"testing"` in `config_by_name` |
| E2E pipeline test | `tests/test_e2e_pipeline.py` | synthetic snapshot → features → model → signal → paper trade → resolution; asserts trade recorded + resolved with correct PnL; Kalshi monkeypatched, no network |
| Page render smoke | `tests/test_pages_render.py` | `client.get` each page route (incl. `/control`) → 200 + key content |
| API integration | `tests/test_api_settings.py`, `tests/test_control_api.py` | `GET/POST /api/settings` round-trip; new control endpoints |
| Baseline data | `quality_baseline.json` | `{"tests_passed": N, "coverage_pct": X}` |
| Ratchet runner | `scripts/check_quality.py` | run `pytest --cov` (JSON), parse pass-count + coverage; **exit 1** if below baseline; on green, ratchet baseline **up** (never down) |
| Auto-run hook | `.claude/settings.json` | `Stop` hook runs `scripts/check_quality.py`; nonzero exit blocks + reports |
| CI (optional) | `.github/workflows/quality.yml` | run `check_quality.py` on push (durability beyond Claude sessions) |

**Ratchet semantics (`check_quality.py`):**
- Runs the full suite with coverage; if any test fails → exit 1.
- If `tests_passed < baseline.tests_passed` or `coverage_pct < baseline.coverage_pct` → exit 1
  (regression = "kill errors immediately").
- On success, if the new numbers exceed the baseline, rewrite `quality_baseline.json` upward.
- `--check-only` mode (used by the hook) never lowers the baseline.

### Piece B — Strategy Control Center

New page `/control` (default landing; `/` redirects to it). Existing pages untouched.

| Component | File | Responsibility |
|---|---|---|
| Page route + redirect + nav | `app/routes/dashboard.py`, `app/templates/base.html` | add `/control` route, `/`→`/control`, nav link |
| Template | `app/templates/control.html` | the Control Center layout (see §5) |
| Focused JS | `app/static/js/control.js` | reads state, renders status strip, wires controls; does NOT touch `main.js` |
| Focused CSS | `app/static/css/control.css` | Control Center styling (reuse base tokens) |
| Apply-defaults endpoint | `app/routes/api.py` | `POST /api/control/apply-defaults` → set validated paper config via `set_setting` |
| Aggregate state endpoint | `app/routes/api.py` | `GET /api/control/state` → mode, scheduler, last-signal, paper P&L today, win-rate-vs-breakeven |

Reuses existing endpoints (`/api/settings`, `/api/scheduler/*`, `/api/live-snapshot`,
`/api/paper/portfolio`) where possible; the two new endpoints are thin.

**Paper-by-default:** update `seed_default_settings()` so fresh deploys come up paper-trading
the validated strategy:
- `paper_trading_enabled="true"`, `auto_trade_enabled="true"`, `scheduler_running="true"`
- `live_trading_enabled="false"` (unchanged), `mispricing_threshold="0.25"` (already)
Covered by an updated seed-defaults test.

## 5. Control Center layout

```
┌──────────────────────────────────────────────────────────────┐
│  STRATEGY CONTROL CENTER                     ●  PAPER MODE      │
├──────────────────────────────────────────────────────────────┤
│  MODE   [ ● PAPER (default) ]  ( LIVE — real money )           │
│         Live is OFF · safe. Flip to LIVE = typed confirmation. │
├──────────────────────────────────────────────────────────────┤
│  STATUS  Scheduler RUNNING · Signal WAITING                    │
│          Last signal 12:03:45 (23s ago) · Trades today 4       │
│          Paper P&L today +$3.20 · Win 71% (breakeven ~67%)     │
├──────────────────────────────────────────────────────────────┤
│  STRATEGY (validated)          [ Apply validated defaults ]    │
│    Signal mode   [ Ensemble ▼ ]                                │
│    Mispricing gap [====|------] 25%                             │
│    Risk profile  [ Moderate ▼ ]  (60–120s)                     │
│    NO calibration gate 20% · Entry caps YES≤65¢ NO≤80¢         │
├──────────────────────────────────────────────────────────────┤
│  SAFETY  Max daily loss $50 · Paper size $10       [ Save ]    │
└──────────────────────────────────────────────────────────────┘
```

Principles: Paper/Live is the single most prominent control (paper default; live needs typed
confirm + red banner); a one-click **Apply validated defaults** sets the whole modern config;
a status strip answers "is it actually paper-trading?"; plain-language labels with one-line
explanations; rarely-touched knobs stay on the old Settings page.

## 6. Testing the new work

- `TestingConfig` + `conftest` fixtures enable clean `create_app("testing")`.
- E2E pipeline test proves the full paper flow.
- Page-render test asserts `/control` returns 200 with "Strategy Control Center".
- Control API tests: `apply-defaults` writes the validated keys; `state` returns the
  expected shape; paper-default verified in the seed-defaults test.
- Everything runs under the ratchet; the Stop hook enforces green + non-regressing coverage.

## 7. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Stop hook slows every turn | Suite is fast (unit + light E2E); acceptable for the guarantee |
| Coverage ratchet flaps on transient dips | Ratchet only raises on green; regression → fix before proceeding (intended) |
| Ephemeral DB resets settings | Seed defaults now = paper-trading validated strategy (safe, not live) |
| Default landing change surprises | `/` → `/control`; other pages still reachable via nav |
| Auto paper-trade on deploy unexpected | Explicit, documented; live stays off; paper only |

## 8. Success criteria

- `pytest` runs via `pyproject.toml` config; `create_app("testing")` works.
- E2E pipeline test passes; page + API integration tests pass.
- `scripts/check_quality.py` fails on regression and ratchets up on success; Stop hook wired.
- `/control` renders, defaults to paper, one-click applies the validated config, status strip
  reflects live state.
- Fresh deploy comes up paper-trading the validated strategy, live off.
- All existing tests still pass; total suite green.
