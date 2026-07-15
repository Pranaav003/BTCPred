"""Run the suite with coverage; fail on regression; ratchet the baseline up on success."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

BASELINE_PATH = Path(__file__).resolve().parent.parent / "quality_baseline.json"


def ratchet(passed: int, coverage_pct: float, baseline: dict,
            allow_raise: bool = True) -> tuple[bool, dict]:
    """Return (ok, new_baseline). ok=False on any regression. Never lowers."""
    if passed < baseline["tests_passed"] or coverage_pct < baseline["coverage_pct"]:
        return False, baseline
    if not allow_raise:
        return True, baseline
    new = {
        "tests_passed": max(passed, baseline["tests_passed"]),
        "coverage_pct": max(coverage_pct, baseline["coverage_pct"]),
    }
    return True, new


def _run_suite() -> tuple[bool, int, float]:
    """Run pytest with coverage. Returns (all_passed, passed_count, coverage_pct)."""
    proc = subprocess.run(
        [".venv/bin/python", "-m", "pytest", "--cov=app", "--cov=sim",
         "--cov-report=term-missing", "-q"],
        capture_output=True, text=True,
    )
    out = proc.stdout + proc.stderr
    # passed count
    m_pass = re.search(r"(\d+) passed", out)
    passed = int(m_pass.group(1)) if m_pass else 0
    # coverage total line: "TOTAL   ...   NN%"
    m_cov = re.search(r"TOTAL\s+.*?(\d+(?:\.\d+)?)%", out)
    coverage = float(m_cov.group(1)) if m_cov else 0.0
    all_passed = proc.returncode == 0
    return all_passed, passed, coverage


def _load_baseline() -> dict:
    if BASELINE_PATH.exists():
        return json.loads(BASELINE_PATH.read_text())
    return {"tests_passed": 0, "coverage_pct": 0.0}


def _save_baseline(baseline: dict) -> None:
    BASELINE_PATH.write_text(json.dumps(baseline, indent=2) + "\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true",
                        help="fail on regression but never raise the baseline")
    parser.add_argument("--init", action="store_true",
                        help="seed the baseline from the current run")
    args = parser.parse_args(argv)

    all_passed, passed, coverage = _run_suite()
    if not all_passed:
        print(f"QUALITY: FAIL — tests not green (passed={passed}, cov={coverage}%)")
        return 1

    if args.init or not BASELINE_PATH.exists():
        _save_baseline({"tests_passed": passed, "coverage_pct": coverage})
        print(f"QUALITY: baseline seeded (tests={passed}, cov={coverage}%)")
        return 0

    baseline = _load_baseline()
    ok, new = ratchet(passed, coverage, baseline, allow_raise=not args.check_only)
    if not ok:
        print(f"QUALITY: FAIL — regression vs baseline {baseline} "
              f"(got tests={passed}, cov={coverage}%)")
        return 1
    if new != baseline:
        _save_baseline(new)
        print(f"QUALITY: PASS — baseline ratcheted up to {new}")
    else:
        print(f"QUALITY: PASS — baseline held at {baseline}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
