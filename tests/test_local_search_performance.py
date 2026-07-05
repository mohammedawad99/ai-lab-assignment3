"""Stage 11-A: delta 2-opt equivalence and candidate-list correctness."""

import random

from src.cvrp.candidate_lists import build_candidate_lists
from src.cvrp.construction import clarke_wright_savings
from src.cvrp.cost import route_cost
from src.cvrp.local_search import (
    improve_route_2opt,
    improve_route_2opt_full,
    relocate_best_improvement_pass,
    two_opt_route_delta,
)
from src.cvrp.model import CVRPInstance
from src.cvrp.validate import validate_solution


def random_case(n_customers, seed):
    """A shuffled route [0, ..., 0] and a Euclidean distance matrix."""
    rng = random.Random(seed)
    points = [(rng.uniform(0, 100), rng.uniform(0, 100))
              for _ in range(n_customers + 1)]
    matrix = [[((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5 for bx, by in points]
              for ax, ay in points]
    perm = list(range(1, n_customers + 1))
    rng.shuffle(perm)
    return [0] + perm + [0], matrix


def small_instance(seed=3, n=12, capacity=40):
    rng = random.Random(seed)
    coordinates = {i: (rng.uniform(0, 100), rng.uniform(0, 100))
                   for i in range(n + 1)}
    demands = {c: float(rng.randint(1, 10)) for c in range(1, n + 1)}
    return CVRPInstance(name="rand", capacity=capacity, vehicle_count=6,
                        coordinates=coordinates, demands=demands, depot_id=0)


# ---------- delta formula ----------

def test_delta_matches_full_cost_difference():
    route, matrix = random_case(15, seed=1)
    base = route_cost(route, matrix)
    n = len(route)
    for i in range(1, n - 2):
        for j in range(i + 1, n - 1):
            a, b, c, d = route[i - 1], route[i], route[j], route[j + 1]
            delta = matrix[a][c] + matrix[b][d] - matrix[a][b] - matrix[c][d]
            reversed_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
            assert abs(delta - (route_cost(reversed_route, matrix) - base)) < 1e-9


def test_best_move_is_an_actual_improvement():
    route, matrix = random_case(20, seed=2)
    move = two_opt_route_delta(route, matrix)
    assert move is not None  # a shuffled route always has crossings
    delta, i, j = move
    improved = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
    assert route_cost(improved, matrix) < route_cost(route, matrix)
    assert abs((route_cost(improved, matrix) - route_cost(route, matrix)) - delta) < 1e-9


# ---------- optimized 2-opt behaviour ----------

def test_never_worsens_route():
    for seed in range(20):
        route, matrix = random_case(random.Random(seed).randint(3, 25), seed)
        improved = improve_route_2opt(route, matrix)
        assert route_cost(improved, matrix) <= route_cost(route, matrix) + 1e-9


def test_matches_old_implementation_on_random_routes():
    for seed in range(60):
        n = random.Random(seed).randint(3, 18)
        route, matrix = random_case(n, seed=100 + seed)
        assert improve_route_2opt(route, matrix) == \
            improve_route_2opt_full(route, matrix)


def test_depot_endpoints_and_customer_multiset_preserved():
    route, matrix = random_case(30, seed=4)
    improved = improve_route_2opt(route, matrix)
    assert improved[0] == 0 and improved[-1] == 0
    assert sorted(improved[1:-1]) == sorted(route[1:-1])


def test_deterministic_under_same_input():
    route, matrix = random_case(25, seed=5)
    assert improve_route_2opt(route, matrix) == improve_route_2opt(route, matrix)
    assert route == [route[0]] + route[1:]  # input not mutated as a list object


def test_short_routes_are_safe():
    _, matrix = random_case(5, seed=6)
    for route in ([], [0], [0, 0], [0, 1, 0], [0, 1, 2, 0]):
        improved = improve_route_2opt(route, matrix)
        assert improved == route          # nothing (or no useful move) to do
        assert improved is not route      # still a copy
    # length 3 customers: first size where a reversal can change the cost
    route, matrix = random_case(3, seed=7)
    improved = improve_route_2opt(route, matrix)
    assert route_cost(improved, matrix) <= route_cost(route, matrix) + 1e-9


# ---------- candidate lists ----------

def test_candidate_lists_size_and_no_self():
    _, matrix = random_case(20, seed=8)
    lists = build_candidate_lists(matrix, k=5)
    assert len(lists) == len(matrix)
    for i, near in enumerate(lists):
        assert len(near) == 5
        assert i not in near


def test_candidate_lists_sorted_by_distance():
    _, matrix = random_case(15, seed=9)
    lists = build_candidate_lists(matrix, k=10)
    for i, near in enumerate(lists):
        distances = [matrix[i][j] for j in near]
        assert distances == sorted(distances)


def test_candidate_lists_stable_tie_breaking():
    # three nodes at equal distance from node 0: ids must come out ascending
    matrix = [[0.0, 5.0, 5.0, 5.0],
              [5.0, 0.0, 1.0, 2.0],
              [5.0, 1.0, 0.0, 1.0],
              [5.0, 2.0, 1.0, 0.0]]
    lists = build_candidate_lists(matrix, k=3)
    assert lists[0] == [1, 2, 3]


def test_candidate_lists_full_fallback():
    _, matrix = random_case(10, seed=10)
    n = len(matrix)
    for k in (None, n - 1, n + 5):
        lists = build_candidate_lists(matrix, k=k)
        assert all(len(near) == n - 1 for near in lists)


def test_candidate_limited_search_still_feasible_and_not_worse():
    from src.cvrp.distance import build_distance_matrix
    instance = small_instance()
    matrix = build_distance_matrix(instance)
    solution = clarke_wright_savings(instance, matrix)
    neighbors = build_candidate_lists(matrix, k=5)

    limited = relocate_best_improvement_pass(instance, solution, matrix,
                                             neighbors=neighbors)
    full = relocate_best_improvement_pass(instance, solution, matrix)
    assert validate_solution(instance, limited).feasible
    assert limited.cost <= solution.cost + 1e-9   # pruning never worsens
    assert full.cost <= limited.cost + 1e-9        # full search at least as good

    route = max(solution.routes, key=len)
    pruned = improve_route_2opt(route, matrix, neighbors=neighbors)
    assert pruned[0] == 0 and pruned[-1] == 0
    assert sorted(pruned[1:-1]) == sorted(route[1:-1])
    assert route_cost(pruned, matrix) <= route_cost(route, matrix) + 1e-9
