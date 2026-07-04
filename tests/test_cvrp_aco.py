"""Tests for the CVRP ACO solver."""

import subprocess
import sys
from pathlib import Path

import pytest

from src.cvrp.io_cvrplib import parse_cvrplib
from src.cvrp.solvers.aco import run_cvrp_aco
from src.cvrp.solvers.result import CVRPSolverResult
from src.cvrp.validate import validate_solution

REPO_ROOT = Path(__file__).resolve().parents[1]
TINY_INSTANCE = REPO_ROOT / "examples" / "tiny_cvrp.vrp"


@pytest.fixture
def tiny():
    return parse_cvrplib(TINY_INSTANCE)


def test_aco_on_tiny(tiny):
    result = run_cvrp_aco(tiny, iterations=10, ants=5, seed=42, timeout_sec=5.0)
    assert isinstance(result, CVRPSolverResult)
    assert result.algorithm == "cvrp_aco"
    assert result.feasible
    assert result.best_cost == pytest.approx(80.64, abs=0.05)
    check = validate_solution(tiny, result.best_solution)
    assert check.feasible, check.errors
    assert result.convergence  # not empty
    assert result.best_cost <= result.initial_cost + 1e-6


def test_aco_reproducible(tiny):
    a = run_cvrp_aco(tiny, iterations=10, ants=5, seed=7, timeout_sec=5.0)
    b = run_cvrp_aco(tiny, iterations=10, ants=5, seed=7, timeout_sec=5.0)
    assert a.best_cost == pytest.approx(b.best_cost)


# ---------- CLI ----------

def run_cli(*extra_args):
    script = REPO_ROOT / "scripts" / "run_cvrp_aco.py"
    return subprocess.run(
        [sys.executable, str(script), "--instance", str(TINY_INSTANCE), *extra_args],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )


def test_cli_runs():
    proc = run_cli("--iterations", "10", "--ants", "5", "--seed", "42", "--timeout", "5")
    assert proc.returncode == 0
    assert "algorithm: cvrp_aco" in proc.stdout
    assert "feasible: yes" in proc.stdout


def test_cli_output_file(tmp_path):
    output = tmp_path / "aco_solution.txt"
    proc = run_cli("--iterations", "10", "--ants", "5", "--seed", "42",
                   "--output", str(output))
    assert proc.returncode == 0
    assert output.exists()
    assert output.read_text().splitlines()[0] == "80.64"
