"""Run the suite with coverage; fail on regression; ratchet the baseline up on success."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = REPO_ROOT / "quality_baseline.json"
VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
VENV_RUFF = REPO_ROOT / ".venv" / "bin" / "ruff"
_SENTINEL = 10**9  # parse-failure count: always treated as a regression, never a silent pass


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


def ratchet_directional(current, baseline_val, direction: str):
    """(ok, moved). up: ok if current>=baseline, moved=max. down: ok if current<=baseline, moved=min."""
    if direction == "up":
        return (current >= baseline_val, max(current, baseline_val))
    return (current <= baseline_val, min(current, baseline_val))


def _run_suite() -> tuple[bool, int, float]:
    """Run pytest with coverage. Returns (all_passed, passed_count, coverage_pct)."""
    proc = subprocess.run(
        [str(VENV_PYTHON), "-m", "pytest", "--cov=app", "--cov=sim",
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


def _run_ruff() -> int:
    """Count ruff violations across app/sim/scripts. Sentinel on unparseable output."""
    proc = subprocess.run(
        [str(VENV_RUFF), "check", "app", "sim", "scripts", "--output-format=concise"],
        capture_output=True, text=True,
    )
    if proc.returncode == 0:
        return 0
    out = proc.stdout + proc.stderr
    m = re.search(r"Found (\d+) error", out)
    if m:
        return int(m.group(1))
    n = len(re.findall(r"^\S+:\d+:\d+:", out, re.M))
    return n if n > 0 else _SENTINEL


def _run_mypy() -> int:
    """Count mypy errors across app/sim. Sentinel on unparseable output."""
    proc = subprocess.run(
        [str(VENV_PYTHON), "-m", "mypy", "app", "sim"],
        capture_output=True, text=True,
    )
    if proc.returncode == 0:
        return 0
    out = proc.stdout + proc.stderr
    m = re.search(r"Found (\d+) error", out)
    if m:
        return int(m.group(1))
    n = len(re.findall(r": error:", out))
    return n if n > 0 else _SENTINEL


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
