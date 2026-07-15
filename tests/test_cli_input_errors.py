"""CLI scripts must report bad input files clearly, never with a traceback."""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# (script, extra args) — every user-facing CVRP CLI entry point
CASES = [
    ("scripts/run_cvrp_baseline.py", []),
    ("scripts/run_cvrp_sa_ts.py", ["--algorithm", "sa"]),
    ("scripts/run_cvrp_aco.py", []),
    ("scripts/run_cvrp_alns.py", []),
    ("scripts/run_cvrp_bnb_lds.py", []),
    ("scripts/run_cvrp_ga_island.py", []),
    ("scripts/run_cvrp_ils.py", []),
    ("scripts/validate_cvrp_solution.py",
     ["--solution", "examples/tiny_solution.txt"]),
]


@pytest.mark.parametrize("script,extra", CASES,
                         ids=[case[0].rsplit("/", 1)[-1] for case in CASES])
def test_missing_instance_file_reports_clean_error(script, extra):
    proc = subprocess.run(
        [sys.executable, script, "--instance", "/nonexistent.vrp"] + extra,
        capture_output=True, text=True, cwd=REPO_ROOT)
    assert proc.returncode != 0
    combined = proc.stdout + proc.stderr
    assert "input error" in combined, combined
    assert "Traceback" not in proc.stderr, proc.stderr
