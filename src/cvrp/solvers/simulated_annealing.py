"""Simulated Annealing for CVRP, starting from the multi-stage baseline."""

import math
import time

import numpy as np

from src.cvrp.baseline import build_multistage_baseline
from src.cvrp.distance import build_distance_matrix
from src.cvrp.model import CVRPInstance, CVRPSolution
from src.cvrp.solvers.neighborhood import clone_routes, random_neighbor
from src.cvrp.solvers.result import CVRPSolverResult
from src.cvrp.validate import validate_solution

CONVERGENCE_EVERY = 50


def run_cvrp_simulated_annealing(instance: CVRPInstance, iterations: int = 1000,
                                 seed: int = 42, timeout_sec: float = 10.0,
                                 initial_temperature: float = 100.0,
                                 cooling_rate: float = 0.995) -> CVRPSolverResult:
    start_elapsed = time.perf_counter()
    start_cpu = time.process_time()

    distance_matrix = build_distance_matrix(instance)
    current = build_multistage_baseline(instance).solution
    initial_cost = current.cost
    best = CVRPSolution(routes=clone_routes(current.routes), cost=current.cost)

    rng = np.random.default_rng(seed)
    temperature = initial_temperature
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

        candidate = random_neighbor(instance, current, distance_matrix, rng)
        if validate_solution(instance, candidate).feasible:
            delta = candidate.cost - current.cost
            if delta <= 0 or (temperature > 1e-12
                              and rng.random() < math.exp(-delta / temperature)):
                current = candidate
            if current.cost < best.cost - 1e-9:
                best = CVRPSolution(routes=clone_routes(current.routes), cost=current.cost)

        temperature *= cooling_rate
        if it % CONVERGENCE_EVERY == 0:
            record(it)

    if not convergence or convergence[-1]["iteration"] != completed:
        record(completed)

    check = validate_solution(instance, best)
    return CVRPSolverResult(
        algorithm="cvrp_sa",
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
