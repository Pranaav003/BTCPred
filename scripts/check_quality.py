"""Run the suite with coverage; fail on regression; ratchet the baseline up on success."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = REPO_ROOT / "test-baseline.json"
VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
VENV_RUFF = REPO_ROOT / ".venv" / "bin" / "ruff"
_SENTINEL = 10**9  # parse-failure count: always treated as a regression, never a silent pass
PERF_TOLERANCE = 1.10
PERF_ABS_FLOOR_MS = 0.5  # absolute floor for sub-ms metrics; +10% alone is noise-tight

_DIRECTIONS = {
    "tests_passed": "up",
    "coverage_pct": "up",
    "ruff_violations": "down",
    "mypy_errors": "down",
}


def ratchet(current: dict, baseline: dict, allow_move: bool = True) -> tuple[bool, dict]:
    """Apply the directional ratchet to all metrics. ok=False if ANY regresses; never moves the wrong way."""
    ok_all = True
    moved = dict(baseline)
    for metric, direction in _DIRECTIONS.items():
        ok, new_val = ratchet_directional(current[metric], baseline[metric], direction)
        ok_all = ok_all and ok
        moved[metric] = new_val
    if not ok_all:
        return False, baseline
    return True, (moved if allow_move else baseline)


def ratchet_directional(current, baseline_val, direction: str):
    """(ok, moved). up: ok if current>=baseline, moved=max. down: ok if current<=baseline, moved=min."""
    if direction == "up":
        return (current >= baseline_val, max(current, baseline_val))
    return (current <= baseline_val, min(current, baseline_val))


def perf_ok(
    current: float, baseline: float,
    tol: float = PERF_TOLERANCE, floor_ms: float = PERF_ABS_FLOOR_MS,
) -> bool:
    # OK if within relative band OR absolute floor (whichever is more generous).
    return current <= max(baseline * tol, baseline + floor_ms)


def perf_ratchet(
    current: dict, baseline: dict, tol: float = PERF_TOLERANCE
) -> tuple[bool, dict]:
    """Perf metrics ratchet DOWN with a +tol band. ok=False if ANY exceeds band.
    Unknown metrics (not in baseline) are accepted and seeded."""
    ok_all = True
    new = dict(baseline)
    for key, cur in current.items():
        base = baseline.get(key)
        if base is None:
            new[key] = cur            # seed a newly-tracked metric
            continue
        if not perf_ok(cur, base, tol):
            ok_all = False
        new[key] = min(cur, base)
    if not ok_all:
        return False, baseline
    return True, new


def _run_perf() -> dict:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "perf_probe", REPO_ROOT / "scripts" / "perf_probe.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.measure()


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


def _measure() -> tuple[bool, dict]:
    all_passed, passed, coverage = _run_suite()
    metrics = {
        "tests_passed": passed,
        "coverage_pct": coverage,
        "ruff_violations": _run_ruff(),
        "mypy_errors": _run_mypy(),
    }
    return all_passed, metrics


def _load_baseline() -> dict:
    if BASELINE_PATH.exists():
        return json.loads(BASELINE_PATH.read_text())
    return {"tests_passed": 0, "coverage_pct": 0.0,
            "ruff_violations": _SENTINEL, "mypy_errors": _SENTINEL}


def _save_baseline(baseline: dict) -> None:
    BASELINE_PATH.write_text(json.dumps(baseline, indent=2) + "\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true",
                        help="fail on regression but never move the baseline")
    parser.add_argument("--init", action="store_true",
                        help="seed the baseline from the current run")
    parser.add_argument("--perf", action="store_true",
                        help="measure hot-path latency; fail on >10% regression")
    args = parser.parse_args(argv)

    if args.perf:
        metrics = _run_perf()
        baseline = _load_baseline()
        ok, new_perf = perf_ratchet(metrics, baseline.get("perf", {}))
        if not ok:
            bl = baseline.get("perf", {})
            print(f"PERF: FAIL — regression. baseline={bl} current={metrics}")
            return 1
        if new_perf != baseline.get("perf"):
            baseline["perf"] = new_perf
            _save_baseline(baseline)
            print(f"PERF: PASS — perf baseline updated to {new_perf}")
        else:
            print(f"PERF: PASS — perf baseline held {new_perf}")
        return 0

    all_passed, metrics = _measure()
    if not all_passed:
        print(f"QUALITY: FAIL — tests not green {metrics}")
        return 1

    if args.init or not BASELINE_PATH.exists():
        to_save = dict(metrics)
        if BASELINE_PATH.exists():
            prev = _load_baseline()
            if "perf" in prev:
                to_save["perf"] = prev["perf"]
        _save_baseline(to_save)
        print(f"QUALITY: baseline seeded {to_save}")
        return 0

    baseline = _load_baseline()
    ok, new = ratchet(metrics, baseline, allow_move=not args.check_only)
    if not ok:
        print(f"QUALITY: FAIL — regression. baseline={baseline} current={metrics}")
        return 1
    if new != baseline:
        _save_baseline(new)
        print(f"QUALITY: PASS — baseline updated to {new}")
    else:
        print(f"QUALITY: PASS — baseline held {baseline}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
