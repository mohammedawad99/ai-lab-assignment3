"""Tabu Search for CVRP, starting from the multi-stage baseline.

A simple foundation: candidates are sampled with the shared random
neighborhood, and the tabu key is the customer sequence of the whole
solution (routes without depot zeros).
"""

import time

import numpy as np

from src.cvrp.baseline import build_multistage_baseline
from src.cvrp.distance import build_distance_matrix
from src.cvrp.model import CVRPInstance, CVRPSolution
from src.cvrp.solvers.neighborhood import clone_routes, random_neighbor
from src.cvrp.solvers.result import CVRPSolverResult
from src.cvrp.validate import validate_solution

CONVERGENCE_EVERY = 50


def solution_signature(solution: CVRPSolution) -> tuple:
    """Tabu key: customer sequences of all routes, depots removed."""
    return tuple(
        tuple(customer for customer in route if customer != 0)
        for route in solution.routes
    )


def run_cvrp_tabu_search(instance: CVRPInstance, iterations: int = 1000,
                         seed: int = 42, timeout_sec: float = 10.0,
                         tabu_tenure: int = 30,
                         candidates_per_iteration: int = 40) -> CVRPSolverResult:
    start_elapsed = time.perf_counter()
    start_cpu = time.process_time()

    distance_matrix = build_distance_matrix(instance)
    current = build_multistage_baseline(instance).solution
    initial_cost = current.cost
    best = CVRPSolution(routes=clone_routes(current.routes), cost=current.cost)

    rng = np.random.default_rng(seed)
    tabu_list: list[tuple] = []  # FIFO, newest at the end
    convergence = []
    completed = 0

    def record(iteration):
        convergence.append({
            "iteration": iteration,
            "best_cost": best.cost,
            "current_cost": current.cost,
            "elapsed_time": time.perf_counter() - start_elapsed,
            "cpu_time": time.process_time() - start_cpu,
        })

    record(0)
    for it in range(1, iterations + 1):
        if time.perf_counter() - start_elapsed > timeout_sec:
            break
        completed = it

        candidates = []
        for _ in range(candidates_per_iteration):
            candidate = random_neighbor(instance, current, distance_matrix, rng)
            if validate_solution(instance, candidate).feasible:
                candidates.append(candidate)
        candidates.sort(key=lambda c: c.cost)

        chosen = None
        for candidate in candidates:
            signature = solution_signature(candidate)
            aspiration = candidate.cost < best.cost - 1e-9
            if signature not in tabu_list or aspiration:
                chosen = candidate
                break

        if chosen is not None:
            current = chosen
            tabu_list.append(solution_signature(chosen))
            if len(tabu_list) > tabu_tenure:
                tabu_list.pop(0)
            if current.cost < best.cost - 1e-9:
                best = CVRPSolution(routes=clone_routes(current.routes), cost=current.cost)

        if it % CONVERGENCE_EVERY == 0:
            record(it)

    if not convergence or convergence[-1]["iteration"] != completed:
        record(completed)

    check = validate_solution(instance, best)
    return CVRPSolverResult(
        algorithm="cvrp_tabu",
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
