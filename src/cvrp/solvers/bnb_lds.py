"""Time-limited Branch and Bound / Limited Discrepancy Search for CVRP.

The search inserts customers one by one (hardest first). At every depth the
feasible insertions are sorted by added cost; taking the cheapest costs 0
discrepancy, the second cheapest 1, and so on. Branches are cut when the
total discrepancy exceeds the limit or when the partial cost already reaches
the incumbent. Starting incumbent is the multi-stage baseline, and the best
feasible incumbent is always returned.

This is exact-inspired and time-limited — NOT a full exact CVRP proof for
the large benchmark instances.
"""

import time

from src.cvrp.baseline import build_multistage_baseline
from src.cvrp.construction import route_load
from src.cvrp.cost import route_cost, solution_cost
from src.cvrp.distance import build_distance_matrix
from src.cvrp.local_search import improve_solution_2opt
from src.cvrp.model import CVRPInstance, CVRPSolution
from src.cvrp.solvers.result import CVRPSolverResult
from src.cvrp.validate import validate_solution

CONVERGENCE_EVERY = 1000


# ---------- helpers ----------

def used_customer_count(routes: list[list[int]]) -> int:
    return sum(1 for route in routes for node in route if node != 0)


def route_count(routes: list[list[int]]) -> int:
    """Number of routes that actually contain customers."""
    return sum(1 for route in routes if any(node != 0 for node in route[1:-1]))


def sort_customers_for_search(instance: CVRPInstance, distance_matrix) -> list[int]:
    """Hard customers first: larger demand, then farther from depot, then id."""
    return sorted(
        instance.customer_ids,
        key=lambda c: (-instance.demands.get(c, 0.0), -distance_matrix[0][c], c),
    )


def partial_solution_cost(routes: list[list[int]], distance_matrix) -> float:
    return sum(route_cost(route, distance_matrix) for route in routes)


def insertion_options(instance: CVRPInstance, routes: list[list[int]],
                      customer: int, distance_matrix) -> list[dict]:
    """Feasible insertions sorted by delta_cost. A new-route option is added
    whenever a vehicle is still free."""
    d = distance_matrix
    demand = instance.demands.get(customer, 0.0)
    options = []
    for r_idx, route in enumerate(routes):
        if route_load(instance, route) + demand > instance.capacity:
            continue
        for pos in range(1, len(route)):
            prev, nxt = route[pos - 1], route[pos]
            options.append({
                "customer": customer,
                "route_index": r_idx,
                "insert_position": pos,
                "delta_cost": d[prev][customer] + d[customer][nxt] - d[prev][nxt],
                "creates_new_route": False,
            })
    if demand <= instance.capacity and route_count(routes) < instance.vehicle_count:
        options.append({
            "customer": customer,
            "route_index": None,
            "insert_position": None,
            "delta_cost": d[0][customer] + d[customer][0],
            "creates_new_route": True,
        })
    options.sort(key=lambda opt: opt["delta_cost"])
    return options


def apply_insertion(routes: list[list[int]], option: dict) -> list[list[int]]:
    """Return new routes with the insertion applied. Input stays untouched."""
    new_routes = [list(route) for route in routes]
    if option["creates_new_route"]:
        new_routes.append([0, option["customer"], 0])
    else:
        new_routes[option["route_index"]].insert(option["insert_position"],
                                                 option["customer"])
    return new_routes


def canonicalize_routes(routes: list[list[int]]) -> list[list[int]]:
    """Copy routes and drop unused [0, 0] ones. No customer is removed."""
    return [list(route) for route in routes if any(node != 0 for node in route[1:-1])]


def routes_to_solution(routes: list[list[int]], distance_matrix) -> CVRPSolution:
    solution = CVRPSolution(routes=[list(route) for route in routes])
    solution.cost = solution_cost(solution, distance_matrix)
    return solution


def greedy_complete(instance: CVRPInstance, routes: list[list[int]],
                    remaining_customers: list[int], distance_matrix):
    """Insert all remaining customers at their cheapest feasible position.
    Returns a full CVRPSolution or None."""
    current = [list(route) for route in routes]
    for customer in remaining_customers:
        options = insertion_options(instance, current, customer, distance_matrix)
        if not options:
            return None
        current = apply_insertion(current, options[0])
    return routes_to_solution(current, distance_matrix)


# ---------- solver ----------

def run_cvrp_bnb_lds(instance: CVRPInstance, max_discrepancy: int = 3,
                     max_nodes: int = 50000, seed: int = 42,
                     timeout_sec: float = 10.0) -> CVRPSolverResult:
    # seed is accepted for interface consistency; the DFS itself is deterministic
    start_elapsed = time.perf_counter()
    start_cpu = time.process_time()

    distance_matrix = build_distance_matrix(instance)
    baseline = build_multistage_baseline(instance).solution
    initial_cost = baseline.cost
    best = CVRPSolution(routes=[list(r) for r in baseline.routes], cost=baseline.cost)

    order = sort_customers_for_search(instance, distance_matrix)
    stats = {
        "expanded_nodes": 0,
        "pruned_by_bound": 0,
        "pruned_by_discrepancy": 0,
        "completed_solutions": 0,
    }
    stop_reason = None
    convergence = []

    def record(current_cost):
        convergence.append({
            "iteration": stats["expanded_nodes"],
            "best_cost": best.cost,
            "current_cost": current_cost,
            "elapsed_time": time.perf_counter() - start_elapsed,
            "cpu_time": time.process_time() - start_cpu,
            **stats,
        })

    def dfs(routes, depth, discrepancy):
        nonlocal best, stop_reason
        if stop_reason is not None:
            return
        if time.perf_counter() - start_elapsed > timeout_sec:
            stop_reason = "timeout"
            return
        if stats["expanded_nodes"] >= max_nodes:
            stop_reason = "max_nodes"
            return
        stats["expanded_nodes"] += 1
        if stats["expanded_nodes"] % CONVERGENCE_EVERY == 0:
            record(partial_solution_cost(routes, distance_matrix))

        if depth == len(order):
            solution = routes_to_solution(canonicalize_routes(routes), distance_matrix)
            solution = improve_solution_2opt(solution, distance_matrix)
            stats["completed_solutions"] += 1
            if (solution.cost < best.cost - 1e-9
                    and validate_solution(instance, solution).feasible):
                best = solution
            return

        options = insertion_options(instance, routes, order[depth], distance_matrix)
        for rank, option in enumerate(options):
            new_discrepancy = discrepancy + rank
            if new_discrepancy > max_discrepancy:
                # options are sorted, so everything from here on is also over the limit
                stats["pruned_by_discrepancy"] += len(options) - rank
                break
            new_routes = apply_insertion(routes, option)
            if partial_solution_cost(new_routes, distance_matrix) >= best.cost - 1e-9:
                stats["pruned_by_bound"] += 1
                continue
            dfs(new_routes, depth + 1, new_discrepancy)
            if stop_reason is not None:
                return

    record(initial_cost)
    dfs([], 0, 0)
    if stop_reason is None:
        stop_reason = "completed"
    record(best.cost)

    # stopping early on timeout/max_nodes is not an error: the incumbent is returned
    check = validate_solution(instance, best)

    return CVRPSolverResult(
        algorithm="cvrp_bnb_lds",
        instance_name=instance.name,
        seed=seed,
        iterations=stats["expanded_nodes"],
        best_solution=best,
        best_cost=best.cost,
        initial_cost=initial_cost,
        feasible=check.feasible,
        errors=check.errors,
        elapsed_time=time.perf_counter() - start_elapsed,
        cpu_time=time.process_time() - start_cpu,
        convergence=convergence,
    )
