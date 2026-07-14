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
