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
