"""Result dataclass shared by the CVRP metaheuristic solvers."""

from dataclasses import dataclass

from src.cvrp.model import CVRPSolution


@dataclass
class CVRPSolverResult:
    algorithm: str
    instance_name: str
    seed: int
    iterations: int
    best_solution: CVRPSolution
    best_cost: float
    initial_cost: float
    feasible: bool
    errors: list[str]
    elapsed_time: float
    cpu_time: float
    # rows like {"iteration": ..., "best_cost": ..., "current_cost": ...,
    #            "elapsed_time": ..., "cpu_time": ...}
    convergence: list[dict]
