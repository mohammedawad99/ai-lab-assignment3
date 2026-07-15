"""Iterated Local Search (ILS) for CVRP.

This is the explicit iterated-local-search driver: it repeatedly kicks the
current local optimum with a perturbation that is deliberately stronger
than any single local-search move, re-optimizes with the deterministic
best-improvement stack, and decides at the ILS level whether to accept the
new local optimum. It is not a rebrand of SA/Tabu/ACO — those solvers walk
one random neighbor at a time, while ILS jumps between complete local
optima. This module intentionally imports no other metaheuristic solver.

Structure of one ILS iteration:
  1. Perturbation (exploration): a ruin-and-recreate kick removes a random
     subset of customers (the kick strength) and greedily reinserts them at
     their cheapest capacity-feasible positions in random order. This moves
     the search away from the current local optimum by rebuilding several
     routes at once — far beyond one relocate/swap/2-opt step.
  2. Local search (exploitation): the strongest deterministic improvement
     stack in the project — intra-route 2-opt plus inter-route relocate,
     swap, Or-opt and 2-opt* (improve_solution_advanced), optionally pruned
     with k-nearest candidate lists on large instances.
  3. Acceptance (at the ILS iteration level): an improved local optimum is
     always accepted; a worse one is accepted only within a small relative
     threshold (threshold acceptance, `worse_tolerance`); after
     `restart_after` iterations without a new global best, the search
     restarts from the best solution and the kick strength grows by one
     (up to a cap) to push exploration further out.

Feasibility: the kick only reuses existing route slots (it never opens a
new route) and reinsertion is capacity-checked, so a feasible input stays
feasible; if the greedy repair cannot place every removed customer, the
kick is rejected and the previous solution is kept. Every accepted
candidate and the final best are validated explicitly.

Complexity: ILS costs O(iterations x (perturbation + local search)).
The perturbation is O(strength x n^2) for the cheapest reinsertion scans;
the local-search stack dominates with roughly O(n^2) per best-improvement
pass (O(1) delta evaluation per candidate move, optionally pruned by
candidate lists). The perturbation adds exploration overhead per
iteration, but it is what lets the search escape local optima that the
deterministic stack alone cannot leave.
"""

import time
from dataclasses import dataclass

import numpy as np

from src.cvrp.baseline import build_multistage_baseline
from src.cvrp.candidate_lists import build_candidate_lists
from src.cvrp.distance import build_distance_matrix
from src.cvrp.local_search import improve_solution_advanced, try_insert_customer_best
from src.cvrp.model import CVRPInstance, CVRPSolution
from src.cvrp.solvers.neighborhood import clone_routes, solution_from_routes
from src.cvrp.solvers.result import CVRPSolverResult
from src.cvrp.validate import validate_solution

EPS = 1e-9


@dataclass
class CVRPILSResult(CVRPSolverResult):
    """CVRPSolverResult plus the ILS-specific counters."""
    accepted_worse_count: int = 0
    perturbation_strength: int = 0
    restarts: int = 0


def ruin_and_recreate_kick(instance: CVRPInstance, solution: CVRPSolution,
                           distance_matrix, rng, strength: int):
    """Perturbation: remove `strength` random customers, then reinsert each
    at its cheapest capacity-feasible position in random order.

    Only existing route slots are reused, so the route count never grows.
    Returns a new CVRPSolution, or None when the greedy repair cannot place
    every removed customer (the caller then rejects the kick safely).
    """
    routes = clone_routes(solution.routes)
    positions = [(i, p) for i, route in enumerate(routes)
                 for p in range(1, len(route) - 1)
                 if route[p] != instance.depot_id]
    if strength < 1 or len(positions) <= strength:
        return None
    picked = rng.choice(len(positions), size=strength, replace=False)
    # delete deepest positions first so earlier indices stay valid
    chosen = sorted((positions[int(k)] for k in picked),
                    key=lambda pos: (pos[0], -pos[1]))
    removed = []
    for route_idx, pos in chosen:
        removed.append(routes[route_idx][pos])
        del routes[route_idx][pos]

    order = list(removed)
    rng.shuffle(order)
    for customer in order:
        result = try_insert_customer_best(instance, routes, customer,
                                          distance_matrix)
        if result is None:
            return None  # repair failed: reject this kick
        routes, _ = result
    return solution_from_routes(routes, distance_matrix)


def default_kick_strength(instance: CVRPInstance) -> int:
    """Base kick size: ~10% of the customers, at least 2, at most 30."""
    return max(2, min(30, round(0.1 * len(instance.customer_ids))))


def run_cvrp_ils(instance: CVRPInstance, iterations: int = 100,
                 seed: int = 42, timeout_sec: float = 10.0,
                 kick_strength: int = None, worse_tolerance: float = 0.02,
                 restart_after: int = 20, max_kick_strength: int = None,
                 local_search_passes: int = 2,
                 candidate_list_k: int = None,
                 candidate_list_min_customers: int = 60) -> CVRPILSResult:
    """Run ILS on one instance. Deterministic under a fixed seed.

    kick_strength defaults to ~10% of the customers; max_kick_strength
    (default: 2x the base) caps the stagnation-driven growth.
    candidate_list_k prunes the local search on instances with at least
    candidate_list_min_customers customers; None keeps the exact scans.
    """
    start_elapsed = time.perf_counter()
    start_cpu = time.process_time()

    distance_matrix = build_distance_matrix(instance)
    neighbors = None
    if (candidate_list_k is not None
            and len(instance.customer_ids) >= candidate_list_min_customers):
        neighbors = build_candidate_lists(distance_matrix, k=candidate_list_k)

    baseline = build_multistage_baseline(instance)
    initial_cost = baseline.solution.cost

    # initial local optimum: exploitation before the first kick
    current = improve_solution_advanced(instance, baseline.solution,
                                        distance_matrix, neighbors=neighbors,
                                        max_passes=local_search_passes)
    best = CVRPSolution(routes=clone_routes(current.routes), cost=current.cost)

    rng = np.random.default_rng(seed)
    base_strength = kick_strength or default_kick_strength(instance)
    max_strength = max_kick_strength or max(base_strength + 2, 2 * base_strength)
    strength = base_strength
    accepted_worse = 0
    restarts = 0
    stagnation = 0
    completed = 0
    convergence = []

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

        # 1. perturbation (exploration)
        kicked = ruin_and_recreate_kick(instance, current, distance_matrix,
                                        rng, strength)
        if kicked is None:
            record(it)
            continue  # kick rejected (repair failed); keep the current optimum

        # 2. local search (exploitation) back to a local optimum
        candidate = improve_solution_advanced(instance, kicked, distance_matrix,
                                              neighbors=neighbors,
                                              max_passes=local_search_passes)
        if not validate_solution(instance, candidate).feasible:
            record(it)
            continue  # never accept an infeasible local optimum

        # 3. acceptance at the ILS level (threshold acceptance)
        delta = candidate.cost - current.cost
        if delta < -EPS:
            current = candidate
        elif candidate.cost <= current.cost * (1.0 + worse_tolerance):
            current = candidate
            if delta > EPS:
                accepted_worse += 1

        if current.cost < best.cost - EPS:
            best = CVRPSolution(routes=clone_routes(current.routes),
                                cost=current.cost)
            stagnation = 0
            strength = base_strength
        else:
            stagnation += 1
            if stagnation >= restart_after:
                # restart acceptance: back to the best local optimum with a
                # stronger kick, so exploration widens instead of looping
                current = CVRPSolution(routes=clone_routes(best.routes),
                                       cost=best.cost)
                strength = min(strength + 1, max_strength)
                restarts += 1
                stagnation = 0
        record(it)

    check = validate_solution(instance, best)
    return CVRPILSResult(
        algorithm="cvrp_ils",
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
        accepted_worse_count=accepted_worse,
        perturbation_strength=strength,
        restarts=restarts,
    )
