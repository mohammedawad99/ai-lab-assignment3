"""Adaptive Large Neighborhood Search for CVRP.

Destroy operators (random removal, worst removal) take customers out of the
solution; repair operators (greedy insertion, regret-2 insertion) put them
back. Operator weights adapt based on how useful each operator was.
"""

import math
import time

import numpy as np

from src.cvrp.baseline import build_multistage_baseline
from src.cvrp.construction import route_load
from src.cvrp.cost import solution_cost
from src.cvrp.distance import build_distance_matrix
from src.cvrp.local_search import improve_solution_2opt
from src.cvrp.model import CVRPInstance, CVRPSolution
from src.cvrp.solvers.result import CVRPSolverResult
from src.cvrp.validate import validate_solution

CONVERGENCE_EVERY = 50
WEIGHT_FLOOR = 0.05
BIG_REGRET = 1e9  # used when a customer has only one feasible insertion

# operator scores
SCORE_NEW_BEST = 5.0
SCORE_IMPROVED = 2.0
SCORE_ACCEPTED = 0.5
SCORE_REJECTED = 0.1


# ---------- small helpers ----------

def copy_solution(solution: CVRPSolution) -> CVRPSolution:
    return CVRPSolution(routes=[list(route) for route in solution.routes],
                        cost=solution.cost)


def solution_customer_set(solution: CVRPSolution) -> set[int]:
    return {node for route in solution.routes for node in route if node != 0}


def choose_by_weight(weights: dict[str, float], rng) -> str:
    names = sorted(weights)
    values = np.array([weights[name] for name in names])
    probs = values / values.sum()
    return names[int(rng.choice(len(names), p=probs))]


def update_weight(weights: dict[str, float], name: str, score: float,
                  reaction_rate: float = 0.2) -> None:
    new_weight = (1.0 - reaction_rate) * weights[name] + reaction_rate * score
    weights[name] = max(WEIGHT_FLOOR, new_weight)


def _used_route_count(routes: list[list[int]]) -> int:
    return sum(1 for route in routes if any(node != 0 for node in route[1:-1]))


# ---------- destroy operators ----------

def random_removal(instance: CVRPInstance, solution: CVRPSolution, rng,
                   remove_count: int):
    """Remove remove_count customers chosen uniformly.
    Returns (partial_solution, removed_customers)."""
    customers = sorted(solution_customer_set(solution))
    remove_count = min(remove_count, len(customers))
    removed = sorted(int(c) for c in rng.choice(customers, size=remove_count,
                                                replace=False))
    removed_set = set(removed)
    routes = [[node for node in route if node == 0 or node not in removed_set]
              for route in solution.routes]
    return CVRPSolution(routes=routes), removed


def worst_removal(instance: CVRPInstance, solution: CVRPSolution,
                  distance_matrix, rng, remove_count: int):
    """Repeatedly remove a customer with a large detour saving,
    picked from the top three to keep some randomness."""
    d = distance_matrix
    routes = [list(route) for route in solution.routes]
    removed = []

    for _ in range(remove_count):
        savings = []  # (saving, route_idx, pos)
        for r_idx, route in enumerate(routes):
            for pos in range(1, len(route) - 1):
                customer = route[pos]
                if customer == 0:
                    continue
                prev, nxt = route[pos - 1], route[pos + 1]
                saving = d[prev][customer] + d[customer][nxt] - d[prev][nxt]
                savings.append((saving, r_idx, pos))
        if not savings:
            break
        savings.sort(key=lambda item: -item[0])
        top = savings[:min(3, len(savings))]
        _, r_idx, pos = top[int(rng.integers(0, len(top)))]
        removed.append(routes[r_idx].pop(pos))

    return CVRPSolution(routes=routes), sorted(removed)


def shaw_related_removal(instance: CVRPInstance, solution: CVRPSolution,
                         distance_matrix, rng, remove_count: int):
    """Shaw-style removal: remove customers that are related to each other
    (geographically close, similar demand, bonus for sharing a route)."""
    d = distance_matrix
    customers = sorted(solution_customer_set(solution))
    if not customers:
        return CVRPSolution(routes=[list(r) for r in solution.routes]), []
    remove_count = min(remove_count, len(customers))

    route_of = {}
    for idx, route in enumerate(solution.routes):
        for node in route:
            if node != 0:
                route_of[node] = idx
    max_distance = max(max(row) for row in d) or 1.0
    max_demand = max(instance.demands.get(c, 0.0) for c in customers) or 1.0

    removed = [int(rng.choice(customers))]
    while len(removed) < remove_count:
        reference = removed[int(rng.integers(0, len(removed)))]
        candidates = [c for c in customers if c not in removed]

        def relatedness(c):
            score = d[reference][c] / max_distance
            score += abs(instance.demands.get(reference, 0.0)
                         - instance.demands.get(c, 0.0)) / max_demand
            if route_of.get(c) != route_of.get(reference):
                score += 0.5  # different route counts as less related
            return score

        candidates.sort(key=lambda c: (relatedness(c), c))
        top = candidates[:min(3, len(candidates))]
        removed.append(int(top[int(rng.integers(0, len(top)))]))

    removed_set = set(removed)
    routes = [[node for node in route if node == 0 or node not in removed_set]
              for route in solution.routes]
    return CVRPSolution(routes=routes), sorted(removed)


def route_removal(instance: CVRPInstance, solution: CVRPSolution,
                  distance_matrix, rng, remove_count: int):
    """Remove one or two complete routes so their customers can be
    redistributed. Other routes are preserved untouched."""
    used = [i for i, route in enumerate(solution.routes)
            if any(node != 0 for node in route[1:-1])]
    if len(used) <= 1:
        return CVRPSolution(routes=[list(r) for r in solution.routes]), []
    average_size = sum(len(solution.routes[i]) - 2 for i in used) / len(used)
    count = 2 if (remove_count > average_size and len(used) > 2) else 1
    picked = {int(i) for i in rng.choice(used, size=count, replace=False)}

    removed = sorted(node for i in picked
                     for node in solution.routes[i] if node != 0)
    routes = [[0, 0] if i in picked else list(route)
              for i, route in enumerate(solution.routes)]
    return CVRPSolution(routes=routes), removed


def segment_removal(instance: CVRPInstance, solution: CVRPSolution,
                    distance_matrix, rng, remove_count: int):
    """Remove a consecutive customer segment from one random used route."""
    used = [i for i, route in enumerate(solution.routes) if len(route) > 2]
    if not used:
        return CVRPSolution(routes=[list(r) for r in solution.routes]), []
    r_idx = int(rng.choice(used))
    route = solution.routes[r_idx]
    inner = route[1:-1]
    segment_length = max(1, min(remove_count, len(inner)))
    start = int(rng.integers(0, len(inner) - segment_length + 1))
    removed = [node for node in inner[start:start + segment_length] if node != 0]

    routes = [list(r) for r in solution.routes]
    routes[r_idx] = route[:1 + start] + route[1 + start + segment_length:]
    return CVRPSolution(routes=routes), sorted(removed)


# ---------- repair operators ----------

def _insertion_options(instance, routes, customer, distance_matrix):
    """All feasible insertions as sorted (delta_cost, route_idx, pos).
    route_idx=None means opening a new route (if a vehicle is free)."""
    d = distance_matrix
    demand = instance.demands.get(customer, 0.0)
    options = []
    for r_idx, route in enumerate(routes):
        if route_load(instance, route) + demand > instance.capacity:
            continue
        for pos in range(1, len(route)):
            prev, nxt = route[pos - 1], route[pos]
            delta = d[prev][customer] + d[customer][nxt] - d[prev][nxt]
            options.append((delta, r_idx, pos))
    if demand <= instance.capacity and _used_route_count(routes) < instance.vehicle_count:
        options.append((d[0][customer] + d[customer][0], None, None))
    options.sort(key=lambda item: item[0])
    return options


def _apply_insertion(routes, customer, option):
    _, r_idx, pos = option
    if r_idx is None:
        routes.append([0, customer, 0])
    else:
        routes[r_idx].insert(pos, customer)


def greedy_repair(instance: CVRPInstance, partial_solution: CVRPSolution,
                  removed_customers: list[int], distance_matrix):
    """Insert every removed customer at its cheapest feasible position.
    Returns a full CVRPSolution or None if some customer cannot be placed."""
    routes = [list(route) for route in partial_solution.routes]
    for customer in removed_customers:
        options = _insertion_options(instance, routes, customer, distance_matrix)
        if not options:
            return None
        _apply_insertion(routes, customer, options[0])
    solution = CVRPSolution(routes=routes)
    solution.cost = solution_cost(solution, distance_matrix)
    return solution


def regret2_repair(instance: CVRPInstance, partial_solution: CVRPSolution,
                   removed_customers: list[int], distance_matrix):
    """Insert the customer with the largest regret (second best - best) first."""
    routes = [list(route) for route in partial_solution.routes]
    remaining = sorted(removed_customers)

    while remaining:
        best_pick = None  # (regret, best_delta, customer, option)
        for customer in remaining:
            options = _insertion_options(instance, routes, customer, distance_matrix)
            if not options:
                return None
            if len(options) == 1:
                regret = BIG_REGRET  # hard customers go in early
            else:
                regret = options[1][0] - options[0][0]
            candidate = (regret, -options[0][0], -customer)
            if best_pick is None or candidate > best_pick[0]:
                best_pick = (candidate, customer, options[0])
        _, customer, option = best_pick
        _apply_insertion(routes, customer, option)
        remaining.remove(customer)

    solution = CVRPSolution(routes=routes)
    solution.cost = solution_cost(solution, distance_matrix)
    return solution


def regret3_repair(instance: CVRPInstance, partial_solution: CVRPSolution,
                   removed_customers: list[int], distance_matrix):
    """Regret-3 insertion: prefer the customer whose third-best insertion is
    much worse than its best one. Falls back to regret-2-like scoring when a
    customer has fewer than three feasible insertions."""
    routes = [list(route) for route in partial_solution.routes]
    remaining = sorted(removed_customers)

    while remaining:
        best_pick = None  # ((regret, -best_delta, -customer), customer, option)
        for customer in remaining:
            options = _insertion_options(instance, routes, customer, distance_matrix)
            if not options:
                return None
            if len(options) >= 3:
                regret = options[2][0] - options[0][0]
            elif len(options) == 2:
                # few placements left: schedule earlier than 3-option customers
                regret = (options[1][0] - options[0][0]) + BIG_REGRET / 2
            else:
                regret = BIG_REGRET
            candidate = (regret, -options[0][0], -customer)
            if best_pick is None or candidate > best_pick[0]:
                best_pick = (candidate, customer, options[0])
        _, customer, option = best_pick
        _apply_insertion(routes, customer, option)
        remaining.remove(customer)

    solution = CVRPSolution(routes=routes)
    solution.cost = solution_cost(solution, distance_matrix)
    return solution


# ---------- full solver ----------

def run_cvrp_alns(instance: CVRPInstance, iterations: int = 1000, seed: int = 42,
                  timeout_sec: float = 10.0, min_removal: int = 1,
                  max_removal_fraction: float = 0.3,
                  initial_temperature: float = 100.0, cooling_rate: float = 0.995,
                  reaction_rate: float = 0.2,
                  enhanced_operators: bool = False) -> CVRPSolverResult:
    start_elapsed = time.perf_counter()
    start_cpu = time.process_time()

    distance_matrix = build_distance_matrix(instance)
    current = build_multistage_baseline(instance).solution
    initial_cost = current.cost
    best = copy_solution(current)

    rng = np.random.default_rng(seed)
    destroy_operators = {
        "random_removal": lambda cur, count: random_removal(instance, cur, rng, count),
        "worst_removal": lambda cur, count: worst_removal(
            instance, cur, distance_matrix, rng, count),
    }
    repair_operators = {
        "greedy_repair": greedy_repair,
        "regret2_repair": regret2_repair,
    }
    if enhanced_operators:
        destroy_operators["shaw_removal"] = lambda cur, count: shaw_related_removal(
            instance, cur, distance_matrix, rng, count)
        destroy_operators["route_removal"] = lambda cur, count: route_removal(
            instance, cur, distance_matrix, rng, count)
        destroy_operators["segment_removal"] = lambda cur, count: segment_removal(
            instance, cur, distance_matrix, rng, count)
        repair_operators["regret3_repair"] = regret3_repair
    destroy_weights = {name: 1.0 for name in destroy_operators}
    repair_weights = {name: 1.0 for name in repair_operators}

    customer_count = len(instance.customer_ids)
    max_removal = max(min_removal, int(max_removal_fraction * customer_count))
    temperature = initial_temperature
    convergence = []
    completed = 0

    def record(iteration):
        row = {
            "iteration": iteration,
            "best_cost": best.cost,
            "current_cost": current.cost,
            "elapsed_time": time.perf_counter() - start_elapsed,
            "cpu_time": time.process_time() - start_cpu,
        }
        for name, weight in destroy_weights.items():
            row[f"{name}_weight"] = weight
        for name, weight in repair_weights.items():
            row[f"{name}_weight"] = weight
        convergence.append(row)

    record(0)
    for it in range(1, iterations + 1):
        if time.perf_counter() - start_elapsed > timeout_sec:
            break
        completed = it

        destroy_name = choose_by_weight(destroy_weights, rng)
        repair_name = choose_by_weight(repair_weights, rng)
        remove_count = int(rng.integers(min_removal, max_removal + 1))

        partial, removed = destroy_operators[destroy_name](current, remove_count)
        candidate = repair_operators[repair_name](instance, partial, removed,
                                                  distance_matrix)

        score = SCORE_REJECTED
        if candidate is not None:
            candidate = improve_solution_2opt(candidate, distance_matrix)
            if validate_solution(instance, candidate).feasible:
                delta = candidate.cost - current.cost
                if delta < -1e-9:
                    current = candidate
                    score = SCORE_IMPROVED
                    if current.cost < best.cost - 1e-9:
                        best = copy_solution(current)
                        score = SCORE_NEW_BEST
                elif rng.random() < math.exp(-delta / temperature):
                    current = candidate
                    score = SCORE_ACCEPTED

        update_weight(destroy_weights, destroy_name, score, reaction_rate)
        update_weight(repair_weights, repair_name, score, reaction_rate)
        temperature = max(temperature * cooling_rate, 1e-12)

        if it % CONVERGENCE_EVERY == 0:
            record(it)

    if not convergence or convergence[-1]["iteration"] != completed:
        record(completed)

    check = validate_solution(instance, best)
    return CVRPSolverResult(
        algorithm="cvrp_alns",
        instance_name=instance.name,
        seed=seed,
        iterations=completed,
        best_solution=best,
        best_cost=best.cost,
        initial_cost=initial_cost,
        feasible=check.feasible,
        errors=check.errors,
        elapsed_time=time.perf_counter() - start_elapsed,
        cpu_time=time.process_time() - start_cpu,
        convergence=convergence,
    )
