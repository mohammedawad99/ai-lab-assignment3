"""Experiment runner for the Ackley warm-up algorithms.

The six required algorithms are sa, tabu, aco, ga_island, alns and bnb_lds.
"random_search" is only a sanity baseline, not one of the six.
"""

from src.ackley.aco import run_ackley_aco
from src.ackley.alns import run_ackley_alns
from src.ackley.bnb_lds import run_ackley_bnb_lds
from src.ackley.ga_island import run_ackley_ga_island
from src.ackley.random_search import run_random_search
from src.ackley.simulated_annealing import run_ackley_simulated_annealing
from src.ackley.tabu_search import run_ackley_tabu_search

ACKLEY_ALGORITHMS = ["random_search", "sa", "tabu", "aco", "ga_island", "alns", "bnb_lds"]

ACKLEY_FIELDNAMES = [
    "part", "algorithm", "seed", "dimension", "lower", "upper",
    "budget_name", "budget_value", "timeout_sec",
    "best_value", "distance_from_origin", "elapsed_time", "cpu_time",
    "iterations", "best_x", "errors",
]


def ackley_result_to_row(result, seed, budget_name, budget_value, timeout_sec) -> dict:
    return {
        "part": "ackley",
        "algorithm": result.algorithm,
        "seed": seed,
        "dimension": result.dimension,
        "lower": "",   # filled in by run_one_ackley_algorithm
        "upper": "",
        "budget_name": budget_name,
        "budget_value": budget_value,
        "timeout_sec": timeout_sec,
        "best_value": f"{result.best_value:.6f}",
        "distance_from_origin": f"{result.distance_from_origin:.6f}",
        "elapsed_time": f"{result.elapsed_time:.6f}",
        "cpu_time": f"{result.cpu_time:.6f}",
        "iterations": result.iterations,
        "best_x": " ".join(f"{v:.4f}" for v in result.best_x),
        "errors": "",
    }


def run_one_ackley_algorithm(algorithm, seed, budget, timeout_sec, dimension=10,
                             lower=-32.768, upper=32.768) -> dict:
    if algorithm == "random_search":
        result = run_random_search(dimension=dimension, iterations=budget,
                                   seed=seed, lower=lower, upper=upper)
        budget_name = "iterations"
    elif algorithm == "sa":
        result = run_ackley_simulated_annealing(dimension=dimension, iterations=budget,
                                                seed=seed, lower=lower, upper=upper,
                                                timeout_sec=timeout_sec)
        budget_name = "iterations"
    elif algorithm == "tabu":
        result = run_ackley_tabu_search(dimension=dimension, iterations=budget,
                                        seed=seed, lower=lower, upper=upper,
                                        timeout_sec=timeout_sec)
        budget_name = "iterations"
    elif algorithm == "aco":
        result = run_ackley_aco(dimension=dimension, iterations=budget, ants=10,
                                seed=seed, lower=lower, upper=upper,
                                bins_per_dimension=10, timeout_sec=timeout_sec)
        budget_name = "iterations"
    elif algorithm == "ga_island":
        result = run_ackley_ga_island(dimension=dimension, generations=budget,
                                      population_size=12, islands=2, seed=seed,
                                      lower=lower, upper=upper,
                                      timeout_sec=timeout_sec)
        budget_name = "generations"
    elif algorithm == "alns":
        result = run_ackley_alns(dimension=dimension, iterations=budget, seed=seed,
                                 lower=lower, upper=upper, timeout_sec=timeout_sec)
        budget_name = "iterations"
    elif algorithm == "bnb_lds":
        result = run_ackley_bnb_lds(dimension=dimension, bins_per_dimension=5,
                                    max_discrepancy=2,
                                    max_nodes=max(1000, budget * 100), seed=seed,
                                    lower=lower, upper=upper,
                                    timeout_sec=timeout_sec)
        budget_name = "max_nodes"
    else:
        raise ValueError(f"unknown Ackley algorithm '{algorithm}'")

    row = ackley_result_to_row(result, seed, budget_name, budget, timeout_sec)
    row["lower"] = lower
    row["upper"] = upper
    return row


def _error_row(algorithm, seed, timeout_sec, dimension, lower, upper, message) -> dict:
    row = {field: "" for field in ACKLEY_FIELDNAMES}
    row.update({
        "part": "ackley",
        "algorithm": algorithm,
        "seed": seed,
        "dimension": dimension,
        "lower": lower,
        "upper": upper,
        "timeout_sec": timeout_sec,
        "errors": message,
    })
    return row


def run_ackley_experiments(algorithms, seeds, budget=100, timeout_sec=10.0,
                           dimension=10, lower=-32.768, upper=32.768) -> list[dict]:
    rows = []
    for algorithm in algorithms:
        for seed in seeds:
            try:
                rows.append(run_one_ackley_algorithm(
                    algorithm, seed, budget, timeout_sec,
                    dimension=dimension, lower=lower, upper=upper,
                ))
            except Exception as exc:
                # one failing run should not kill the whole experiment
                rows.append(_error_row(algorithm, seed, timeout_sec,
                                       dimension, lower, upper, str(exc)))
    return rows
