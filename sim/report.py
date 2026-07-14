"""Render search results into a leaderboard CSV and a markdown summary."""
from __future__ import annotations

import os

import pandas as pd


def leaderboard_to_dataframe(board: list) -> pd.DataFrame:
    rows = []
    for r in board:
        cfg, tm = r["config"], r["test_metrics"]
        rows.append({
            "signal": cfg.get("signal"),
            "exit": cfg.get("exit"),
            "sizing": cfg.get("sizing"),
            "passed": r["passed"],
            "score": r["score"],
            "test_n_trades": tm.get("n_trades"),
            "test_win_rate": tm.get("win_rate"),
            "test_total_pnl": tm.get("total_pnl"),
            "test_profit_factor": tm.get("profit_factor"),
            "test_max_drawdown": tm.get("max_drawdown"),
            "mc_pvalue": r["mc_pvalue"],
            "wf_positive_folds": r["wf_positive_folds"],
        })
    return pd.DataFrame(rows)


def write_report(board: list, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    df = leaderboard_to_dataframe(board)
    csv_path = os.path.join(out_dir, "leaderboard.csv")
    md_path = os.path.join(out_dir, "summary.md")
    df.to_csv(csv_path, index=False)

    passed = [r for r in board if r["passed"]]
    lines = ["# Strategy Search Summary", ""]
    if passed:
        top = passed[0]
        cfg = top["config"]
        tm = top["test_metrics"]
        lines += [
            f"**Configs evaluated:** {len(board)} — **passed all gates:** {len(passed)}",
            "",
            "## Top passing strategy",
            f"- signal=`{cfg.get('signal')}` exit=`{cfg.get('exit')}` "
            f"sizing=`{cfg.get('sizing')}`",
            f"- test P&L: {tm.get('total_pnl'):.2f} | win rate: "
            f"{tm.get('win_rate'):.1%} | profit factor: {tm.get('profit_factor'):.2f}",
            f"- max drawdown: {tm.get('max_drawdown'):.2f} | "
            f"MC p-value: {top['mc_pvalue']:.3f} | "
            f"walk-forward positive folds: {top['wf_positive_folds']}",
        ]
    else:
        lines += [
            f"**Configs evaluated:** {len(board)}",
            "",
            "## No strategy passed all gates",
            "No configuration cleared the out-of-sample, walk-forward, "
            "Monte-Carlo, drawdown, and trade-count gates. This is a valid "
            "negative result: on this data, no robust edge was found.",
        ]
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return {"csv": csv_path, "md": md_path, "n_passed": len(passed)}
