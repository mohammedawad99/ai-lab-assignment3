"""Tests for the explicit CVRP Iterated Local Search (Stage 13-A)."""

from pathlib import Path

import pytest

from src.cvrp.construction import route_load
from src.cvrp.distance import build_distance_matrix
from src.cvrp.io_cvrplib import parse_cvrplib
from src.cvrp.solvers.ils import (
    CVRPILSResult,
    default_kick_strength,
    ruin_and_recreate_kick,
    run_cvrp_ils,
)
from src.cvrp.validate import validate_solution

REPO_ROOT = Path(__file__).resolve().parents[1]
TINY = REPO_ROOT / "examples" / "tiny_cvrp.vrp"
ILS_SOURCE = REPO_ROOT / "src" / "cvrp" / "solvers" / "ils.py"


@pytest.fixture()
def tiny_instance():
    return parse_cvrplib(TINY)


def test_ils_module_exists_and_imports():
    assert ILS_SOURCE.exists()
    assert callable(run_cvrp_ils)
    assert callable(ruin_and_recreate_kick)


def test_ils_returns_feasible_solution_on_tiny(tiny_instance):
    result = run_cvrp_ils(tiny_instance, iterations=15, seed=42, timeout_sec=5.0)
    assert result.feasible
    assert not result.errors
    assert validate_solution(tiny_instance, result.best_solution).feasible
    # tiny instance optimum is ~80.64 and ILS must reach it
    assert result.best_cost == pytest.approx(80.6449510224598, abs=0.01)


def test_ils_routes_use_depot_format_and_capacity(tiny_instance):
    result = run_cvrp_ils(tiny_instance, iterations=10, seed=42, timeout_sec=5.0)
    for route in result.best_solution.routes:
        assert route[0] == tiny_instance.depot_id
        assert route[-1] == tiny_instance.depot_id
        assert route_load(tiny_instance, route) <= tiny_instance.capacity


def test_ils_visits_every_customer_exactly_once(tiny_instance):
    result = run_cvrp_ils(tiny_instance, iterations=10, seed=42, timeout_sec=5.0)
    visited = [node for route in result.best_solution.routes
               for node in route[1:-1] if node != tiny_instance.depot_id]
    assert sorted(visited) == sorted(tiny_instance.customer_ids)


def test_ils_is_deterministic_under_fixed_seed(tiny_instance):
    a = run_cvrp_ils(tiny_instance, iterations=25, seed=7, timeout_sec=5.0)
    b = run_cvrp_ils(tiny_instance, iterations=25, seed=7, timeout_sec=5.0)
    assert a.best_cost == b.best_cost
    assert a.best_solution.routes == b.best_solution.routes
    assert a.accepted_worse_count == b.accepted_worse_count
    assert a.restarts == b.restarts


def test_ils_exposes_elapsed_and_cpu_time(tiny_instance):
    result = run_cvrp_ils(tiny_instance, iterations=10, seed=42, timeout_sec=5.0)
    assert isinstance(result, CVRPILSResult)
    assert result.elapsed_time > 0.0
    assert result.cpu_time > 0.0
    assert result.convergence  # per-iteration progress with both clocks
    assert {"elapsed_time", "cpu_time"} <= set(result.convergence[0])


def test_perturbation_is_a_real_multi_customer_kick(tiny_instance):
    """The kick must move several customers at once — a stronger step than
    any single local-search move — while preserving feasibility."""
    import numpy as np

    matrix = build_distance_matrix(tiny_instance)
    from src.cvrp.baseline import build_multistage_baseline
    start = build_multistage_baseline(tiny_instance).solution

    rng = np.random.default_rng(123)
    kicked = ruin_and_recreate_kick(tiny_instance, start, matrix, rng, strength=2)
    assert kicked is not None
    assert validate_solution(tiny_instance, kicked).feasible
    before = [node for route in start.routes for node in route[1:-1]]
    after = [node for route in kicked.routes for node in route[1:-1]]
    assert sorted(before) == sorted(after)  # no customer lost or duplicated

    # asking to remove more customers than exist is rejected, not crashed
    too_strong = ruin_and_recreate_kick(tiny_instance, start, matrix, rng,
                                        strength=len(before) + 1)
    assert too_strong is None


def test_default_kick_strength_scales_with_instance(tiny_instance):
    assert default_kick_strength(tiny_instance) == 2  # 4 customers -> minimum


def test_ils_does_not_wrap_another_metaheuristic():
    """ILS must be its own driver, not a rebranded SA/TS/ACO/ALNS run."""
    source = ILS_SOURCE.read_text(encoding="utf-8")
    for forbidden in ["solvers.simulated_annealing", "solvers.tabu_search",
                      "solvers.aco", "solvers.alns", "solvers.ga_island",
                      "solvers.bnb_lds"]:
        assert forbidden not in source, forbidden
    # and it must contain the real ILS ingredients
    assert "ruin_and_recreate_kick" in source
    assert "improve_solution_advanced" in source
    assert "worse_tolerance" in source


def test_ils_registered_in_experiment_runner(tiny_instance):
    from src.experiments.cvrp_experiments import (
        CVRP_ALGORITHMS,
        run_one_cvrp_algorithm,
    )
    assert "ils" in CVRP_ALGORITHMS
    row = run_one_cvrp_algorithm(tiny_instance, TINY, "ils", seed=42,
                                 budget=10, timeout_sec=5.0)
    assert row["algorithm"] == "cvrp_ils"
    assert row["feasible"] is True
    assert float(row["best_cost"]) > 0
    assert float(row["elapsed_time"]) > 0
    assert float(row["cpu_time"]) > 0


def test_ils_smoke_budget_is_fast(tiny_instance):
    result = run_cvrp_ils(tiny_instance, iterations=3, seed=42, timeout_sec=2.0)
    assert result.feasible
    assert result.iterations <= 3
    assert result.elapsed_time < 2.5
