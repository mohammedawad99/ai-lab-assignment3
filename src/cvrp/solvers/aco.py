"""Ant Colony Optimization for CVRP.

Each ant builds a capacity-aware solution edge by edge, guided by pheromone
and inverse distance. Pheromone evaporates every iteration and is deposited
on the edges of the iteration-best and global-best solutions.
"""

import time

import numpy as np

from src.cvrp.baseline import build_multistage_baseline
from src.cvrp.distance import build_distance_matrix
from src.cvrp.local_search import improve_solution_2opt
from src.cvrp.model import CVRPInstance, CVRPSolution
from src.cvrp.solvers.neighborhood import clone_routes, solution_from_routes
from src.cvrp.solvers.result import CVRPSolverResult
from src.cvrp.validate import validate_solution

CONVERGENCE_EVERY = 10
EPS = 1e-9
PHEROMONE_FLOOR = 1e-6


def _construct_ant_solution(instance, distance_matrix, pheromone, heuristic,
                            rng, alpha, beta):
    """One ant builds a full solution. Returns None if it runs out of vehicles."""
    depot = instance.depot_id
    unvisited = set(instance.customer_ids)
    routes = []

    for _ in range(instance.vehicle_count):
        if not unvisited:
            break
        route = [depot]
        current = depot
        remaining = instance.capacity
        while True:
            feasible = sorted(c for c in unvisited
                              if instance.demands.get(c, 0.0) <= remaining)
            if not feasible:
                break
            weights = np.array([
                (pheromone[current][c] ** alpha) * (heuristic[current][c] ** beta)
                for c in feasible
            ])
            total = float(weights.sum())
            if total <= 0 or not np.all(np.isfinite(weights)):
                chosen = int(rng.choice(feasible))
            else:
                chosen = int(rng.choice(feasible, p=weights / total))
            route.append(chosen)
            unvisited.remove(chosen)
            remaining -= instance.demands.get(chosen, 0.0)
            current = chosen
        route.append(depot)
        if len(route) > 2:
            routes.append(route)

    if unvisited:
        return None  # not all customers fit into the available vehicles
    return solution_from_routes(routes, distance_matrix)


def _deposit(pheromone, solution, amount):
    for route in solution.routes:
        for a, b in zip(route, route[1:]):
            pheromone[a][b] += amount
            pheromone[b][a] += amount


def run_cvrp_aco(instance: CVRPInstance, iterations: int = 100, ants: int = 20,
                 seed: int = 42, timeout_sec: float = 10.0, alpha: float = 1.0,
                 beta: float = 2.0, evaporation_rate: float = 0.2,
                 pheromone_deposit: float = 100.0) -> CVRPSolverResult:
    start_elapsed = time.perf_counter()
    start_cpu = time.process_time()

    distance_matrix = build_distance_matrix(instance)
    n = instance.node_count
    heuristic = [[1.0 / (distance_matrix[i][j] + EPS) for j in range(n)] for i in range(n)]
    pheromone = [[1.0] * n for _ in range(n)]

    baseline = build_multistage_baseline(instance).solution
    initial_cost = baseline.cost
    best = CVRPSolution(routes=clone_routes(baseline.routes), cost=baseline.cost)

    rng = np.random.default_rng(seed)
    convergence = []
    completed = 0

    def record(iteration, current_cost):
        convergence.append({
            "iteration": iteration,
            "best_cost": best.cost,
            "current_cost": current_cost,
            "elapsed_time": time.perf_counter() - start_elapsed,
            "cpu_time": time.process_time() - start_cpu,
        })

    def out_of_time():
        return time.perf_counter() - start_elapsed > timeout_sec

    record(0, best.cost)
    for it in range(1, iterations + 1):
        if out_of_time():
            break
        completed = it

        iteration_best = None
        for _ in range(ants):
            if out_of_time():
                break
            solution = _construct_ant_solution(
                instance, distance_matrix, pheromone, heuristic, rng, alpha, beta
            )
            if solution is None:
                continue
            solution = improve_solution_2opt(solution, distance_matrix)
            if not validate_solution(instance, solution).feasible:
                continue
            if iteration_best is None or solution.cost < iteration_best.cost:
                iteration_best = solution

        if iteration_best is not None and iteration_best.cost < best.cost - 1e-9:
            best = CVRPSolution(routes=clone_routes(iteration_best.routes),
                                cost=iteration_best.cost)

        # evaporation, with a small floor to keep values positive
        for i in range(n):
            for j in range(n):
                pheromone[i][j] = max(pheromone[i][j] * (1.0 - evaporation_rate),
                                      PHEROMONE_FLOOR)
        # deposit on the iteration best and (half strength) on the global best
        if iteration_best is not None:
            _deposit(pheromone, iteration_best, pheromone_deposit / iteration_best.cost)
        _deposit(pheromone, best, 0.5 * pheromone_deposit / best.cost)

        if it % CONVERGENCE_EVERY == 0:
            record(it, iteration_best.cost if iteration_best else best.cost)

    if not convergence or convergence[-1]["iteration"] != completed:
        record(completed, best.cost)

    check = validate_solution(instance, best)
    return CVRPSolverResult(
        algorithm="cvrp_aco",
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
