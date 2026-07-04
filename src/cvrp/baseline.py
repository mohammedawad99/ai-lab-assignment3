"""Multi-stage CVRP baseline: construction -> 2-opt -> relocate -> validation.

This is the explicit multi-stage heuristic required by the assignment. It will
later also provide the starting incumbent for the metaheuristic solvers.
"""

from dataclasses import dataclass

from src.cvrp.construction import clarke_wright_savings
from src.cvrp.distance import build_distance_matrix
from src.cvrp.local_search import improve_solution_2opt, relocate_best_improvement_pass
from src.cvrp.model import CVRPInstance, CVRPSolution
from src.cvrp.validate import validate_solution


@dataclass
class BaselineResult:
    solution: CVRPSolution
    construction_cost: float
    after_2opt_cost: float
    final_cost: float
    feasible: bool
    errors: list[str]


def build_multistage_baseline(instance: CVRPInstance) -> BaselineResult:
    distance_matrix = build_distance_matrix(instance)

    # stage 1: Clarke-Wright savings construction
    solution = clarke_wright_savings(instance, distance_matrix)
    construction_cost = solution.cost

    # stage 2: 2-opt inside each route
    solution = improve_solution_2opt(solution, distance_matrix)
    after_2opt_cost = solution.cost

    # stage 3: relocate customers between routes
    solution = relocate_best_improvement_pass(instance, solution, distance_matrix)
    final_cost = solution.cost

    # stage 4: final validation
    check = validate_solution(instance, solution)

    return BaselineResult(
        solution=solution,
        construction_cost=construction_cost,
        after_2opt_cost=after_2opt_cost,
        final_cost=final_cost,
        feasible=check.feasible,
        errors=check.errors,
    )
