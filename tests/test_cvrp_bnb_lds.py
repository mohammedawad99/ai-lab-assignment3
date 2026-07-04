"""Tests for the CVRP B&B/LDS solver."""

import subprocess
import sys
from pathlib import Path

import pytest

from src.cvrp.distance import build_distance_matrix
from src.cvrp.io_cvrplib import parse_cvrplib
from src.cvrp.solvers.bnb_lds import (
    apply_insertion,
    greedy_complete,
    insertion_options,
    run_cvrp_bnb_lds,
    sort_customers_for_search,
)
from src.cvrp.solvers.result import CVRPSolverResult
from src.cvrp.validate import validate_solution

REPO_ROOT = Path(__file__).resolve().parents[1]
TINY_INSTANCE = REPO_ROOT / "examples" / "tiny_cvrp.vrp"


@pytest.fixture
def tiny():
    return parse_cvrplib(TINY_INSTANCE)


# ---------- helpers ----------

def test_sort_customers_for_search(tiny):
    matrix = build_distance_matrix(tiny)
    order = sort_customers_for_search(tiny, matrix)
    assert sorted(order) == sorted(tiny.customer_ids)
    # demand-5 customers (3, 4) come first; 3 is farther from the depot
    assert order[:2] == [3, 4]


def test_insertion_options_on_empty_routes(tiny):
    matrix = build_distance_matrix(tiny)
    options = insertion_options(tiny, [], 1, matrix)
    assert len(options) >= 1
    assert options[0]["creates_new_route"]
    assert options[0]["delta_cost"] == pytest.approx(2 * matrix[0][1])


def test_apply_insertion_does_not_mutate(tiny):
    matrix = build_distance_matrix(tiny)
    routes = [[0, 3, 0]]
    options = insertion_options(tiny, routes, 4, matrix)
    new_routes = apply_insertion(routes, options[0])
    assert routes == [[0, 3, 0]]
    assert new_routes != routes


def test_greedy_complete(tiny):
    matrix = build_distance_matrix(tiny)
    order = sort_customers_for_search(tiny, matrix)
    solution = greedy_complete(tiny, [], order, matrix)
    assert solution is not None
    check = validate_solution(tiny, solution)
    assert check.feasible, check.errors


# ---------- solver ----------

def test_bnb_lds_on_tiny(tiny):
    result = run_cvrp_bnb_lds(tiny, max_discrepancy=3, max_nodes=10000,
                              seed=42, timeout_sec=5.0)
    assert isinstance(result, CVRPSolverResult)
    assert result.algorithm == "cvrp_bnb_lds"
    assert result.feasible
    assert result.best_cost == pytest.approx(80.64, abs=0.05)
    assert result.best_cost <= result.initial_cost + 1e-6
    assert result.convergence
    for key in ("expanded_nodes", "pruned_by_bound", "pruned_by_discrepancy",
                "completed_solutions"):
        assert key in result.convergence[-1]


def test_bnb_lds_max_nodes_safety(tiny):
    result = run_cvrp_bnb_lds(tiny, max_discrepancy=3, max_nodes=1,
                              seed=42, timeout_sec=5.0)
    # even with almost no search, the baseline incumbent is returned
    assert result.feasible
    assert result.best_cost == pytest.approx(80.64, abs=0.05)
    assert result.iterations <= 1


def test_bnb_lds_reproducible(tiny):
    a = run_cvrp_bnb_lds(tiny, max_discrepancy=3, max_nodes=10000, seed=7,
                         timeout_sec=5.0)
    b = run_cvrp_bnb_lds(tiny, max_discrepancy=3, max_nodes=10000, seed=7,
                         timeout_sec=5.0)
    assert a.best_cost == pytest.approx(b.best_cost)
    assert a.iterations == b.iterations


# ---------- CLI ----------

def run_cli(*extra_args):
    script = REPO_ROOT / "scripts" / "run_cvrp_bnb_lds.py"
    return subprocess.run(
        [sys.executable, str(script), "--instance", str(TINY_INSTANCE), *extra_args],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )


def test_cli_runs():
    proc = run_cli("--max-discrepancy", "3", "--max-nodes", "10000",
                   "--seed", "42", "--timeout", "5")
    assert proc.returncode == 0
    assert "algorithm: cvrp_bnb_lds" in proc.stdout
    assert "feasible: yes" in proc.stdout
    assert "final search stats:" in proc.stdout


def test_cli_output_file(tmp_path):
    output = tmp_path / "bnb_lds_solution.txt"
    proc = run_cli("--max-discrepancy", "3", "--max-nodes", "10000",
                   "--seed", "42", "--output", str(output))
    assert proc.returncode == 0
    assert output.exists()
    assert output.read_text().splitlines()[0] == "80.64"
