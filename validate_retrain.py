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

    brier_ok = (old_brier is not None) and (new_brier < old_brier)
    ev_ok = n > 0 and ev > 0
    if brier_ok and ev_ok:
        print("GATE: PASS")
        return 0
    print(f"GATE: FAIL (brier_ok={brier_ok}, ev_ok={ev_ok})")
    return 1


if __name__ == "__main__":
    sys.exit(main())
