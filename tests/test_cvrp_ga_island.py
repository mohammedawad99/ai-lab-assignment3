"""Tests for the CVRP GA Island Model."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.cvrp.distance import build_distance_matrix
from src.cvrp.io_cvrplib import parse_cvrplib
from src.cvrp.solvers.ga_island import (
    chromosome_from_solution,
    inversion_mutation,
    make_initial_population,
    migrate_ring,
    ordered_crossover,
    random_chromosome,
    run_cvrp_ga_island,
    split_giant_tour,
    swap_mutation,
)
from src.cvrp.solvers.result import CVRPSolverResult
from src.cvrp.validate import validate_solution

REPO_ROOT = Path(__file__).resolve().parents[1]
TINY_INSTANCE = REPO_ROOT / "examples" / "tiny_cvrp.vrp"


@pytest.fixture
def tiny():
    return parse_cvrplib(TINY_INSTANCE)


# ---------- representation ----------

def test_split_giant_tour_feasible(tiny):
    matrix = build_distance_matrix(tiny)
    solution = split_giant_tour(tiny, [1, 2, 3, 4], matrix)
    assert solution is not None
    check = validate_solution(tiny, solution)
    assert check.feasible, check.errors
    assert solution.cost == pytest.approx(80.64, abs=0.05)


def test_chromosome_from_solution(tiny):
    matrix = build_distance_matrix(tiny)
    solution = split_giant_tour(tiny, [1, 2, 3, 4], matrix)
    chromosome = chromosome_from_solution(solution)
    assert sorted(chromosome) == [1, 2, 3, 4]


def test_random_chromosome_is_permutation(tiny):
    rng = np.random.default_rng(0)
    for _ in range(10):
        assert sorted(random_chromosome(tiny, rng)) == [1, 2, 3, 4]


def test_ordered_crossover_is_permutation(tiny):
    rng = np.random.default_rng(1)
    p1, p2 = [1, 2, 3, 4], [4, 3, 2, 1]
    for _ in range(20):
        child = ordered_crossover(p1, p2, rng)
        assert sorted(child) == [1, 2, 3, 4]
    assert p1 == [1, 2, 3, 4] and p2 == [4, 3, 2, 1]  # parents untouched


def test_mutations_preserve_permutation(tiny):
    rng = np.random.default_rng(2)
    original = [1, 2, 3, 4]
    for _ in range(20):
        assert sorted(swap_mutation(original, rng, mutation_rate=1.0)) == [1, 2, 3, 4]
        assert sorted(inversion_mutation(original, rng, mutation_rate=1.0)) == [1, 2, 3, 4]
    assert original == [1, 2, 3, 4]


# ---------- population and migration ----------

def test_make_initial_population(tiny):
    matrix = build_distance_matrix(tiny)
    rng = np.random.default_rng(3)
    population = make_initial_population(tiny, matrix, 10, rng)
    assert len(population) == 10
    assert all(sorted(c) == [1, 2, 3, 4] for c in population)


def test_migrate_ring_keeps_sizes(tiny):
    matrix = build_distance_matrix(tiny)
    rng = np.random.default_rng(4)
    islands = [make_initial_population(tiny, matrix, 6, rng) for _ in range(3)]
    migrated = migrate_ring(islands, matrix, tiny, migrants=2)
    assert len(migrated) == 3
    assert all(len(island) == 6 for island in migrated)


# ---------- full solver ----------

def test_ga_island_on_tiny(tiny):
    result = run_cvrp_ga_island(tiny, generations=10, population_size=10,
                                islands=2, seed=42, timeout_sec=5.0)
    assert isinstance(result, CVRPSolverResult)
    assert result.algorithm == "cvrp_ga_island"
    assert result.feasible
    assert result.best_cost == pytest.approx(80.64, abs=0.05)
    assert result.convergence  # not empty
    assert result.best_cost <= result.initial_cost + 1e-6


def test_ga_island_reproducible(tiny):
    a = run_cvrp_ga_island(tiny, generations=10, population_size=10,
                           islands=2, seed=7, timeout_sec=5.0)
    b = run_cvrp_ga_island(tiny, generations=10, population_size=10,
                           islands=2, seed=7, timeout_sec=5.0)
    assert a.best_cost == pytest.approx(b.best_cost)


# ---------- CLI ----------

def run_cli(*extra_args):
    script = REPO_ROOT / "scripts" / "run_cvrp_ga_island.py"
    return subprocess.run(
        [sys.executable, str(script), "--instance", str(TINY_INSTANCE), *extra_args],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )


def test_cli_runs():
    proc = run_cli("--generations", "10", "--population-size", "10",
                   "--islands", "2", "--seed", "42", "--timeout", "5")
    assert proc.returncode == 0
    assert "algorithm: cvrp_ga_island" in proc.stdout
    assert "feasible: yes" in proc.stdout


def test_cli_output_file(tmp_path):
    output = tmp_path / "ga_island_solution.txt"
    proc = run_cli("--generations", "10", "--population-size", "10",
                   "--islands", "2", "--seed", "42", "--output", str(output))
    assert proc.returncode == 0
    assert output.exists()
    assert output.read_text(encoding="utf-8").splitlines()[0] == "80.64"


def test_population_larger_than_permutation_count_terminates(tiny):
    # tiny has 4 customers = only 24 distinct permutations; asking for a
    # population of 30 used to loop forever in make_initial_population
    matrix = build_distance_matrix(tiny)
    rng = np.random.default_rng(0)
    population = make_initial_population(tiny, matrix, 30, rng)
    assert len(population) == 30
    assert all(sorted(c) == [1, 2, 3, 4] for c in population)


def test_tuned_ga_settings_terminate_on_tiny(tiny):
    result = run_cvrp_ga_island(tiny, generations=10, population_size=30,
                                islands=2, mutation_rate=0.3, seed=42,
                                timeout_sec=5.0)
    assert result.feasible
    assert result.best_cost == pytest.approx(80.64, abs=0.05)
