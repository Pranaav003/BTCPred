"""End-to-end pipeline tests.

Two coverage goals:
  1. Model inference smoke: predict_proba_raw(feature_dict) -> float in [0, 1].
     Skipped when raw_feature_model.pkl is absent.

  2. DB-backed pipeline: place a paper trade through the real execute_paper_trade
     path, resolve it via the real resolve_paper_trades helper, and assert the
     resulting realized_pnl sign is correct for the outcome.

No network calls — Kalshi helpers are monkeypatched to return None.
"""

import os
import pytest

MODEL_PRESENT = os.path.exists("raw_feature_model.pkl")


@pytest.mark.e2e
@pytest.mark.skipif(not MODEL_PRESENT, reason="raw_feature_model.pkl artifact required")
def test_model_inference_returns_probability():
    """predict_proba_raw returns a float in [0, 1] with a minimal feature dict."""
    from app.model_loader import predict_proba_raw

    # Missing features default to 0.0 inside predict_proba_raw.
    p = predict_proba_raw({"seconds_to_close": 100, "return_1m": 5.0})
    assert isinstance(p, float)
    assert 0.0 <= p <= 1.0


@pytest.mark.e2e
def test_paper_trade_executes_and_resolves(app, monkeypatch):
    """DB-backed pipeline slice: place a paper trade, resolve it, assert PnL sign.

    Strategy:
      - Seed a Market row and a Signal row with p_market=0.40.
      - Monkeypatch get_active_market / get_live_snapshot to return None so
        execute_paper_trade falls through to the DB Signal for pricing.
      - Execute a NO position (entry_price = 1 - 0.40 = 0.60).
      - Resolve the market with final_outcome_yes=False (NO wins).
      - Assert realized_pnl > 0.
    """
    from datetime import datetime, timezone, timedelta

    from app.models import Market, PaperTrade
    from app.extensions import db
    from app.db_helpers import resolve_paper_trades, get_or_create_market, save_signal

    # Monkeypatch external Kalshi calls so no network traffic occurs.
    import app.paper_trading as _pt_module
    monkeypatch.setattr(_pt_module, "get_active_market", lambda: None)
    monkeypatch.setattr(_pt_module, "get_live_snapshot", lambda: None)

    from app.paper_trading import execute_paper_trade

    ticker = "KXBTC15M-TEST0000"

    # Seed Market row (past close so it's already "closed" for resolution).
    close_time = datetime.now(timezone.utc) - timedelta(minutes=5)
    market = get_or_create_market(
        ticker=ticker,
        title="E2E test market",
        close_time=close_time,
        series_ticker="KXBTC15M",
    )

    # Seed a Signal row with p_market=0.40 so execute_paper_trade can price the NO.
    save_signal(
        market=market,
        snapshot_dict={
            "snapshot_ts": int(close_time.timestamp()),
            "seconds_to_close": 300,
            "entry_bucket": 2,
            "p_market": 0.40,
        },
        signal_str="PAPER BUY NO",
        reason_str="e2e test signal",
        agreement_region_str="test",
        p_raw=0.10,
        yes_cutoff=0.65,
        no_cutoff=0.35,
        raw_features_dict={},
    )

    # Execute a NO paper trade for 1 contract.
    result = execute_paper_trade(
        side="no",
        ticker=ticker,
        contracts=1,
        signal_triggered=False,
    )

    assert isinstance(result, dict), f"execute_paper_trade returned: {result}"
    assert result.get("success") is True, f"Trade not placed: {result}"

    stored = PaperTrade.query.filter_by(ticker=ticker).first()
    assert stored is not None
    assert stored.resolved is False
    assert stored.side == "NO"

    # Verify entry price is the complement of p_market (0.60 = 1 - 0.40).
    assert abs(stored.entry_price - 0.60) < 1e-6, f"Unexpected entry_price: {stored.entry_price}"

    # --- Resolution ---
    # Set market outcome to NO (final_outcome_yes=False) and call the real helper.
    market.resolved = True
    market.final_outcome_yes = False
    db.session.commit()

    resolved_count = resolve_paper_trades(market)
    assert resolved_count == 1

    db.session.refresh(stored)
    assert stored.resolved is True
    # NO wins when outcome is NO: pnl = (1.0 - 0.60) * 1 = 0.40 > 0
    assert stored.realized_pnl is not None
    assert stored.realized_pnl > 0, (
        f"Expected positive PnL for winning NO trade, got {stored.realized_pnl}"
    )
    assert stored.outcome_correct is True
