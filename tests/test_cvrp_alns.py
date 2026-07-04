"""Tests for the CVRP ALNS solver."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.cvrp.baseline import build_multistage_baseline
from src.cvrp.distance import build_distance_matrix
from src.cvrp.io_cvrplib import parse_cvrplib
from src.cvrp.solvers.alns import (
    choose_by_weight,
    greedy_repair,
    random_removal,
    regret2_repair,
    run_cvrp_alns,
    solution_customer_set,
    update_weight,
    worst_removal,
)
from src.cvrp.solvers.result import CVRPSolverResult
from src.cvrp.validate import validate_solution

REPO_ROOT = Path(__file__).resolve().parents[1]
TINY_INSTANCE = REPO_ROOT / "examples" / "tiny_cvrp.vrp"


@pytest.fixture
def tiny():
    return parse_cvrplib(TINY_INSTANCE)


@pytest.fixture
def tiny_setup(tiny):
    matrix = build_distance_matrix(tiny)
    solution = build_multistage_baseline(tiny).solution
    return tiny, matrix, solution


# ---------- destroy operators ----------

def test_random_removal(tiny_setup):
    instance, matrix, solution = tiny_setup
    rng = np.random.default_rng(0)
    partial, removed = random_removal(instance, solution, rng, remove_count=2)
    assert len(removed) == 2
    assert len(set(removed)) == 2
    assert solution_customer_set(partial).isdisjoint(removed)
    assert solution_customer_set(partial) | set(removed) == set(instance.customer_ids)
    # input solution untouched
    assert solution_customer_set(solution) == set(instance.customer_ids)


def test_worst_removal(tiny_setup):
    instance, matrix, solution = tiny_setup
    rng = np.random.default_rng(1)
    partial, removed = worst_removal(instance, solution, matrix, rng, remove_count=2)
    assert len(removed) >= 1
    assert len(set(removed)) == len(removed)
    assert solution_customer_set(partial).isdisjoint(removed)
    assert solution_customer_set(partial) | set(removed) == set(instance.customer_ids)


# ---------- repair operators ----------

@pytest.mark.parametrize("repair", [greedy_repair, regret2_repair])
def test_repair_rebuilds_full_solution(tiny_setup, repair):
    instance, matrix, solution = tiny_setup
    rng = np.random.default_rng(2)
    partial, removed = random_removal(instance, solution, rng, remove_count=2)
    repaired = repair(instance, partial, removed, matrix)
    assert repaired is not None
    check = validate_solution(instance, repaired)
    assert check.feasible, check.errors
    assert solution_customer_set(repaired) == set(instance.customer_ids)


# ---------- adaptive weights ----------

def test_choose_by_weight():
    rng = np.random.default_rng(3)
    weights = {"a": 1.0, "b": 5.0}
    for _ in range(10):
        assert choose_by_weight(weights, rng) in weights


def test_update_weight():
    weights = {"a": 1.0}
    update_weight(weights, "a", score=5.0, reaction_rate=0.2)
    assert weights["a"] == pytest.approx(1.8)
    for _ in range(50):
        update_weight(weights, "a", score=0.0, reaction_rate=0.9)
    assert weights["a"] >= 0.05  # floor keeps it positive


# ---------- full solver ----------

def test_alns_on_tiny(tiny):
    result = run_cvrp_alns(tiny, iterations=20, seed=42, timeout_sec=5.0)
    assert isinstance(result, CVRPSolverResult)
    assert result.algorithm == "cvrp_alns"
    assert result.feasible
    assert result.best_cost == pytest.approx(80.64, abs=0.05)
    assert result.best_cost <= result.initial_cost + 1e-6
    assert result.convergence
    for key in ("random_removal_weight", "worst_removal_weight",
                "greedy_repair_weight", "regret2_repair_weight"):
        assert key in result.convergence[-1]


def test_alns_reproducible(tiny):
    a = run_cvrp_alns(tiny, iterations=20, seed=7, timeout_sec=5.0)
    b = run_cvrp_alns(tiny, iterations=20, seed=7, timeout_sec=5.0)
    assert a.best_cost == pytest.approx(b.best_cost)


# ---------- CLI ----------

def run_cli(*extra_args):
    script = REPO_ROOT / "scripts" / "run_cvrp_alns.py"
    return subprocess.run(
        [sys.executable, str(script), "--instance", str(TINY_INSTANCE), *extra_args],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )


def test_cli_runs():
    proc = run_cli("--iterations", "20", "--seed", "42", "--timeout", "5")
    assert proc.returncode == 0
    assert "algorithm: cvrp_alns" in proc.stdout
    assert "feasible: yes" in proc.stdout
    assert "final operator weights:" in proc.stdout


def test_cli_output_file(tmp_path):
    output = tmp_path / "alns_solution.txt"
    proc = run_cli("--iterations", "20", "--seed", "42", "--output", str(output))
    assert proc.returncode == 0
    assert output.exists()
    assert output.read_text().splitlines()[0] == "80.64"


# ---------- stage 10-A: enhanced operators ----------

def test_shaw_removal_never_removes_depot(tiny_setup):
    from src.cvrp.solvers.alns import shaw_related_removal
    instance, matrix, solution = tiny_setup
    rng = np.random.default_rng(5)
    partial, removed = shaw_related_removal(instance, solution, matrix, rng,
                                            remove_count=2)
    assert len(removed) == 2
    assert 0 not in removed
    assert solution_customer_set(partial).isdisjoint(removed)
    assert solution_customer_set(partial) | set(removed) == set(instance.customer_ids)


def test_route_removal_preserves_other_routes(tiny_setup):
    from src.cvrp.solvers.alns import route_removal
    instance, matrix, solution = tiny_setup
    rng = np.random.default_rng(6)
    partial, removed = route_removal(instance, solution, matrix, rng, remove_count=1)
    assert removed  # at least one full route was emptied
    assert 0 not in removed
    # non-removed routes must be untouched
    survivors = [r for r in partial.routes if len(r) > 2]
    for route in survivors:
        assert route in solution.routes
    assert solution_customer_set(partial) | set(removed) == set(instance.customer_ids)


def test_segment_removal_keeps_all_customers(tiny_setup):
    from src.cvrp.solvers.alns import segment_removal
    instance, matrix, solution = tiny_setup
    rng = np.random.default_rng(7)
    partial, removed = segment_removal(instance, solution, matrix, rng, remove_count=2)
    assert removed
    assert 0 not in removed
    assert solution_customer_set(partial) | set(removed) == set(instance.customer_ids)


def test_regret3_repair_rebuilds_full_solution(tiny_setup):
    from src.cvrp.solvers.alns import regret3_repair
    instance, matrix, solution = tiny_setup
    rng = np.random.default_rng(8)
    partial, removed = random_removal(instance, solution, rng, remove_count=2)
    repaired = regret3_repair(instance, partial, removed, matrix)
    assert repaired is not None
    check = validate_solution(instance, repaired)
    assert check.feasible, check.errors
    assert solution_customer_set(repaired) == set(instance.customer_ids)


def test_enhanced_alns_feasible_on_tiny(tiny):
    result = run_cvrp_alns(tiny, iterations=30, seed=42, timeout_sec=5.0,
                           enhanced_operators=True)
    assert result.feasible
    assert result.best_cost == pytest.approx(80.64, abs=0.05)


def test_enhanced_destroy_repair_roundtrip_no_loss(tiny_setup):
    # many random destroy+repair cycles must never lose or duplicate customers
    from src.cvrp.solvers.alns import (regret3_repair, route_removal,
                                       segment_removal, shaw_related_removal)
    instance, matrix, solution = tiny_setup
    rng = np.random.default_rng(9)
    expected = set(instance.customer_ids)
    for destroy in (shaw_related_removal, route_removal, segment_removal):
        current = solution
        for _ in range(10):
            partial, removed = destroy(instance, current, matrix, rng, 2)
            repaired = regret3_repair(instance, partial, removed, matrix)
            assert repaired is not None
            assert solution_customer_set(repaired) == expected
            current = repaired


def test_tuning_script_help():
    import subprocess, sys
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "tune_cvrp_algorithms.py"),
         "--help"], capture_output=True, text=True, cwd=REPO_ROOT)
    assert proc.returncode == 0
    assert "--algorithms" in proc.stdout
