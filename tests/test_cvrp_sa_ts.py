"""Tests for the CVRP SA and Tabu Search foundations."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.cvrp.baseline import build_multistage_baseline
from src.cvrp.distance import build_distance_matrix
from src.cvrp.io_cvrplib import parse_cvrplib
from src.cvrp.solvers.neighborhood import random_neighbor
from src.cvrp.solvers.result import CVRPSolverResult
from src.cvrp.solvers.simulated_annealing import run_cvrp_simulated_annealing
from src.cvrp.solvers.tabu_search import run_cvrp_tabu_search
from src.cvrp.validate import validate_solution

REPO_ROOT = Path(__file__).resolve().parents[1]
TINY_INSTANCE = REPO_ROOT / "examples" / "tiny_cvrp.vrp"


@pytest.fixture
def tiny():
    return parse_cvrplib(TINY_INSTANCE)


# ---------- neighborhood ----------

def test_random_neighbor_stays_feasible(tiny):
    matrix = build_distance_matrix(tiny)
    solution = build_multistage_baseline(tiny).solution
    rng = np.random.default_rng(0)
    for _ in range(50):
        neighbor = random_neighbor(tiny, solution, matrix, rng)
        check = validate_solution(tiny, neighbor)
        assert check.feasible, check.errors
        solution = neighbor  # walk around the neighborhood a bit


def test_random_neighbor_keeps_all_customers(tiny):
    matrix = build_distance_matrix(tiny)
    solution = build_multistage_baseline(tiny).solution
    rng = np.random.default_rng(1)
    expected = sorted(tiny.customer_ids)
    for _ in range(50):
        neighbor = random_neighbor(tiny, solution, matrix, rng)
        visited = sorted(n for route in neighbor.routes for n in route if n != 0)
        assert visited == expected
        solution = neighbor


# ---------- simulated annealing ----------

def test_sa_on_tiny(tiny):
    result = run_cvrp_simulated_annealing(tiny, iterations=200, seed=42, timeout_sec=5.0)
    assert isinstance(result, CVRPSolverResult)
    assert result.algorithm == "cvrp_sa"
    assert result.feasible
    assert result.best_cost == pytest.approx(80.64, abs=0.05)
    assert result.best_cost <= result.initial_cost + 1e-6
    assert len(result.convergence) >= 2


def test_sa_reproducible(tiny):
    a = run_cvrp_simulated_annealing(tiny, iterations=150, seed=7, timeout_sec=5.0)
    b = run_cvrp_simulated_annealing(tiny, iterations=150, seed=7, timeout_sec=5.0)
    assert a.best_cost == pytest.approx(b.best_cost)


# ---------- tabu search ----------

def test_tabu_on_tiny(tiny):
    result = run_cvrp_tabu_search(tiny, iterations=100, seed=42, timeout_sec=5.0)
    assert result.algorithm == "cvrp_tabu"
    assert result.feasible
    assert result.best_cost == pytest.approx(80.64, abs=0.05)
    assert result.best_cost <= result.initial_cost + 1e-6


def test_tabu_reproducible(tiny):
    a = run_cvrp_tabu_search(tiny, iterations=60, seed=7, timeout_sec=5.0)
    b = run_cvrp_tabu_search(tiny, iterations=60, seed=7, timeout_sec=5.0)
    assert a.best_cost == pytest.approx(b.best_cost)


# ---------- CLI ----------

def run_cli(*extra_args):
    script = REPO_ROOT / "scripts" / "run_cvrp_sa_ts.py"
    return subprocess.run(
        [sys.executable, str(script), "--instance", str(TINY_INSTANCE), *extra_args],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )


def test_cli_sa():
    proc = run_cli("--algorithm", "sa", "--iterations", "100", "--seed", "42")
    assert proc.returncode == 0
    assert "algorithm: cvrp_sa" in proc.stdout
    assert "feasible: yes" in proc.stdout


def test_cli_tabu():
    proc = run_cli("--algorithm", "tabu", "--iterations", "50", "--seed", "42")
    assert proc.returncode == 0
    assert "algorithm: cvrp_tabu" in proc.stdout
    assert "feasible: yes" in proc.stdout


def test_cli_output_file(tmp_path):
    output = tmp_path / "sa_solution.txt"
    proc = run_cli("--algorithm", "sa", "--iterations", "50", "--seed", "42",
                   "--output", str(output))
    assert proc.returncode == 0
    assert output.exists()
    assert output.read_text().splitlines()[0] == "80.64"
