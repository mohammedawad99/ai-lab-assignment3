"""Stage 11-B: advanced CVRP moves (swap, Or-opt, 2-opt*) and the combined
advanced improvement pass, plus solver integration smoke tests."""

import random

from src.cvrp.candidate_lists import build_candidate_lists
from src.cvrp.construction import clarke_wright_savings, route_load
from src.cvrp.distance import build_distance_matrix
from src.cvrp.local_search import (
    improve_solution_advanced,
    or_opt_pass,
    relocate_best_improvement_pass,
    swap_best_improvement_pass,
    two_opt_star_pass,
)
from src.cvrp.model import CVRPInstance, CVRPSolution
from src.cvrp.solvers.alns import run_cvrp_alns
from src.cvrp.solvers.ga_island import run_cvrp_ga_island
from src.cvrp.validate import validate_solution


def make_instance(seed=11, n=16, capacity=30, vehicles=6):
    rng = random.Random(seed)
    coordinates = {i: (rng.uniform(0, 100), rng.uniform(0, 100))
                   for i in range(n + 1)}
    demands = {c: float(rng.randint(2, 9)) for c in range(1, n + 1)}
    return CVRPInstance(name="rand-adv", capacity=capacity,
                        vehicle_count=vehicles, coordinates=coordinates,
                        demands=demands, depot_id=0)


def start_solution(instance, matrix):
    solution = clarke_wright_savings(instance, matrix)
    assert validate_solution(instance, solution).feasible
    return solution


def customer_multiset(solution):
    return sorted(node for route in solution.routes
                  for node in route if node != 0)


def loads_ok(instance, solution):
    return all(route_load(instance, route) <= instance.capacity + 1e-9
               for route in solution.routes)


INSTANCE = make_instance()
MATRIX = build_distance_matrix(INSTANCE)
START = start_solution(INSTANCE, MATRIX)


# ---------- single passes ----------

def test_relocate_preserves_multiset_and_capacity():
    result = relocate_best_improvement_pass(INSTANCE, START, MATRIX)
    assert customer_multiset(result) == customer_multiset(START)
    assert loads_ok(INSTANCE, result)
    assert result.cost <= START.cost + 1e-9


def test_relocate_does_not_apply_without_improvement():
    once = relocate_best_improvement_pass(INSTANCE, START, MATRIX)
    twice = relocate_best_improvement_pass(INSTANCE, once, MATRIX)
    assert abs(twice.cost - once.cost) < 1e-9


def test_swap_preserves_multiset_and_capacity():
    result = swap_best_improvement_pass(INSTANCE, START, MATRIX)
    assert customer_multiset(result) == customer_multiset(START)
    assert loads_ok(INSTANCE, result)
    assert result.cost <= START.cost + 1e-9
    assert validate_solution(INSTANCE, result).feasible


def test_or_opt_preserves_multiset_and_capacity():
    result = or_opt_pass(INSTANCE, START, MATRIX)
    assert customer_multiset(result) == customer_multiset(START)
    assert loads_ok(INSTANCE, result)
    assert result.cost <= START.cost + 1e-9
    assert validate_solution(INSTANCE, result).feasible


def test_or_opt_safe_on_short_routes():
    solution = CVRPSolution(routes=[[0, 1, 0], [0, 2, 0], [0, 0]])
    small = make_instance(n=2, capacity=30)
    matrix = build_distance_matrix(small)
    result = or_opt_pass(small, solution, matrix, segment_lengths=(2, 3))
    assert customer_multiset(result) == [1, 2]


def test_two_opt_star_preserves_multiset_and_capacity():
    result = two_opt_star_pass(INSTANCE, START, MATRIX)
    assert customer_multiset(result) == customer_multiset(START)
    assert loads_ok(INSTANCE, result)
    assert result.cost <= START.cost + 1e-9
    assert validate_solution(INSTANCE, result).feasible


# ---------- combined advanced pass ----------

def test_advanced_never_worsens_and_stays_feasible():
    for seed in (1, 2, 3, 4):
        instance = make_instance(seed=seed, n=14, capacity=25)
        matrix = build_distance_matrix(instance)
        solution = start_solution(instance, matrix)
        improved = improve_solution_advanced(instance, solution, matrix,
                                             validate_each_pass=True)
        assert improved.cost <= solution.cost + 1e-9
        assert customer_multiset(improved) == customer_multiset(solution)
        assert validate_solution(instance, improved).feasible


def test_advanced_deterministic():
    a = improve_solution_advanced(INSTANCE, START, MATRIX)
    b = improve_solution_advanced(INSTANCE, START, MATRIX)
    assert a.routes == b.routes
    assert abs(a.cost - b.cost) < 1e-12


def test_advanced_with_candidate_lists_all_k():
    full = improve_solution_advanced(INSTANCE, START, MATRIX)
    for k in (2, 5, 10):
        neighbors = build_candidate_lists(MATRIX, k=k)
        pruned = improve_solution_advanced(INSTANCE, START, MATRIX,
                                           neighbors=neighbors,
                                           validate_each_pass=True)
        assert customer_multiset(pruned) == customer_multiset(START)
        assert pruned.cost <= START.cost + 1e-9
        # pruning can only reduce the move set, never beat the full search
        # by more than floating noise on this tiny instance
        assert pruned.cost >= full.cost - 1e-6


def test_advanced_flags_disable_moves():
    only_2opt = improve_solution_advanced(
        INSTANCE, START, MATRIX, enable_relocate=False, enable_swap=False,
        enable_or_opt=False, enable_two_opt_star=False)
    assert customer_multiset(only_2opt) == customer_multiset(START)
    assert only_2opt.cost <= START.cost + 1e-9


# ---------- solver integration smoke tests ----------

def test_alns_advanced_smoke():
    result = run_cvrp_alns(INSTANCE, iterations=40, seed=42, timeout_sec=5.0,
                           advanced_local_search=True, advanced_every=10,
                           candidate_list_k=8)
    assert result.feasible
    assert result.best_cost <= result.initial_cost + 1e-9
    repeat = run_cvrp_alns(INSTANCE, iterations=40, seed=42, timeout_sec=5.0,
                           advanced_local_search=True, advanced_every=10,
                           candidate_list_k=8)
    assert abs(repeat.best_cost - result.best_cost) < 1e-9  # deterministic


def test_ga_advanced_smoke():
    result = run_cvrp_ga_island(INSTANCE, generations=20, population_size=10,
                                islands=2, seed=42, timeout_sec=5.0,
                                advanced_local_search=True,
                                local_search_every=5, candidate_list_k=8)
    assert result.feasible
    assert result.best_cost <= result.initial_cost + 1e-9


def test_size_gate_keeps_small_instances_on_current_settings():
    # the tuned config only turns the advanced pass on from min_customers up,
    # so a small instance must produce exactly the non-advanced result
    from src.experiments.cvrp_experiments import run_one_cvrp_algorithm
    tuned = {"alns_advanced": {"advanced_local_search": True,
                               "candidate_list_k": 10, "min_customers": 60}}
    gated = run_one_cvrp_algorithm(INSTANCE, "unused-path", "alns", 42,
                                   30, 5.0, tuned=tuned)
    plain = run_one_cvrp_algorithm(INSTANCE, "unused-path", "alns", 42,
                                   30, 5.0, tuned=None)
    assert gated["best_cost"] == plain["best_cost"]  # 16 customers < 60
    tuned["alns_advanced"]["min_customers"] = 5
    active = run_one_cvrp_algorithm(INSTANCE, "unused-path", "alns", 42,
                                    30, 5.0, tuned=tuned)
    assert str(active["feasible"]).lower() == "true"
