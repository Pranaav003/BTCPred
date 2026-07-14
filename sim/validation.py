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
