"""Experiment runner for the CVRP algorithms.

The six required algorithms are sa, tabu, aco, ga_island, alns and bnb_lds.
"baseline" is only the multi-stage heuristic reference, not one of the six.
"""

from pathlib import Path

from src.common.timing import measure_time
from src.cvrp.baseline import build_multistage_baseline
from src.cvrp.distance import build_distance_matrix
from src.cvrp.io_cvrplib import parse_cvrplib
from src.cvrp.solution_io import format_solution
from src.cvrp.solvers.aco import run_cvrp_aco
from src.cvrp.solvers.alns import run_cvrp_alns
from src.cvrp.solvers.bnb_lds import run_cvrp_bnb_lds
from src.cvrp.solvers.ga_island import run_cvrp_ga_island
from src.cvrp.solvers.simulated_annealing import run_cvrp_simulated_annealing
from src.cvrp.solvers.tabu_search import run_cvrp_tabu_search

CVRP_ALGORITHMS = ["baseline", "sa", "tabu", "aco", "ga_island", "alns", "bnb_lds"]

CVRP_FIELDNAMES = [
    "part", "instance", "instance_path", "algorithm", "seed",
    "budget_name", "budget_value", "timeout_sec",
    "best_cost", "initial_cost", "bks_cost", "gap_percent",
    "feasible", "errors", "elapsed_time", "cpu_time", "iterations",
    "solution_path",
]


def _gap_percent(best_cost, bks_cost) -> str:
    if bks_cost is not None and bks_cost > 0:
        return f"{100.0 * (best_cost - bks_cost) / bks_cost:.4f}"
    return ""


def _write_solution_file(instance, solution, output_dir, algorithm, seed,
                         include_unused_vehicles) -> Path:
    solutions_dir = Path(output_dir) / "cvrp_solutions"
    solutions_dir.mkdir(parents=True, exist_ok=True)
    matrix = build_distance_matrix(instance)
    text = format_solution(solution, matrix,
                           include_unused_vehicles=include_unused_vehicles,
                           vehicle_count=instance.vehicle_count)
    path = solutions_dir / f"{instance.name}_{algorithm}_seed{seed}.txt"
    path.write_text(text + "\n")
    return path


def cvrp_result_to_row(result, instance_path, seed, budget_name, budget_value,
                       timeout_sec, solution_path=None, bks_cost=None) -> dict:
    return {
        "part": "cvrp",
        "instance": result.instance_name,
        "instance_path": str(instance_path),
        "algorithm": result.algorithm,
        "seed": seed,
        "budget_name": budget_name,
        "budget_value": budget_value,
        "timeout_sec": timeout_sec,
        "best_cost": f"{result.best_cost:.4f}",
        "initial_cost": f"{result.initial_cost:.4f}",
        "bks_cost": "" if bks_cost is None else bks_cost,
        "gap_percent": _gap_percent(result.best_cost, bks_cost),
        "feasible": result.feasible,
        "errors": " | ".join(result.errors),
        "elapsed_time": f"{result.elapsed_time:.6f}",
        "cpu_time": f"{result.cpu_time:.6f}",
        "iterations": result.iterations,
        "solution_path": "" if solution_path is None else str(solution_path),
    }


def run_one_cvrp_algorithm(instance, instance_path, algorithm, seed, budget,
                           timeout_sec, output_dir=None,
                           include_unused_vehicles=False, bks_cost=None,
                           tuned=None) -> dict:
    """tuned is the optional dict from configs/tuned_cvrp_settings.json;
    without it every algorithm runs with the original Stage 7-A settings."""
    tuned = tuned or {}
    if algorithm == "baseline":
        baseline, timer = measure_time(build_multistage_baseline, instance)
        solution_path = None
        if output_dir is not None:
            solution_path = _write_solution_file(
                instance, baseline.solution, output_dir, algorithm, seed,
                include_unused_vehicles,
            )
        return {
            "part": "cvrp",
            "instance": instance.name,
            "instance_path": str(instance_path),
            "algorithm": "baseline",
            "seed": seed,
            "budget_name": "none",
            "budget_value": "",
            "timeout_sec": timeout_sec,
            "best_cost": f"{baseline.final_cost:.4f}",
            "initial_cost": f"{baseline.construction_cost:.4f}",
            "bks_cost": "" if bks_cost is None else bks_cost,
            "gap_percent": _gap_percent(baseline.final_cost, bks_cost),
            "feasible": baseline.feasible,
            "errors": " | ".join(baseline.errors),
            "elapsed_time": f"{timer.elapsed_time:.6f}",
            "cpu_time": f"{timer.cpu_time:.6f}",
            "iterations": "",
            "solution_path": "" if solution_path is None else str(solution_path),
        }

    if algorithm == "sa":
        sa_cfg = tuned.get("sa", {})
        result = run_cvrp_simulated_annealing(
            instance,
            iterations=budget * int(sa_cfg.get("iterations_factor", 1)),
            seed=seed, timeout_sec=timeout_sec,
            initial_temperature=sa_cfg.get("initial_temperature", 100.0),
            cooling_rate=sa_cfg.get("cooling_rate", 0.995))
        budget_name = "iterations"
    elif algorithm == "tabu":
        result = run_cvrp_tabu_search(instance, iterations=budget, seed=seed,
                                      timeout_sec=timeout_sec)
        budget_name = "iterations"
    elif algorithm == "aco":
        result = run_cvrp_aco(instance, iterations=budget, ants=10, seed=seed,
                              timeout_sec=timeout_sec)
        budget_name = "iterations"
    elif algorithm == "ga_island":
        ga_cfg = tuned.get("ga_island", {})
        # Stage 11-B advanced pass, gated by instance size (not by name):
        # validation showed gains on large instances and a small regression
        # on the tiny ones, so it only activates from min_customers up
        ga_advanced = bool(ga_cfg.get("advanced_local_search", False)) and \
            len(instance.customer_ids) >= int(ga_cfg.get("advanced_min_customers", 0))
        result = run_cvrp_ga_island(
            instance, generations=budget,
            population_size=int(ga_cfg.get("population_size", 12)),
            islands=int(ga_cfg.get("islands", 2)),
            mutation_rate=ga_cfg.get("mutation_rate", 0.15),
            seed=seed, timeout_sec=timeout_sec,
            advanced_local_search=ga_advanced,
            local_search_every=int(ga_cfg.get("local_search_every", 10)),
            advanced_max_passes=int(ga_cfg.get("advanced_max_passes", 1)),
            candidate_list_k=ga_cfg.get("candidate_list_k"))
        budget_name = "generations"
    elif algorithm in ("alns", "alns_enhanced"):
        adv_cfg = tuned.get("alns_advanced", {})
        alns_advanced = bool(adv_cfg.get("advanced_local_search", False)) and \
            len(instance.customer_ids) >= int(adv_cfg.get("min_customers", 0))
        kwargs = dict(
            advanced_local_search=alns_advanced,
            advanced_every=int(adv_cfg.get("advanced_every", 25)),
            advanced_max_passes=int(adv_cfg.get("advanced_max_passes", 2)),
            candidate_list_k=adv_cfg.get("candidate_list_k"))
        if algorithm == "alns_enhanced":
            alns_cfg = tuned.get("alns", {})
            kwargs.update(enhanced_operators=True,
                          reaction_rate=alns_cfg.get("reaction_rate", 0.2))
        result = run_cvrp_alns(instance, iterations=budget, seed=seed,
                               timeout_sec=timeout_sec, **kwargs)
        budget_name = "iterations"
    elif algorithm == "bnb_lds":
        bnb_cfg = tuned.get("bnb_lds", {})
        max_discrepancy = 3
        max_nodes = max(1000, budget * 100)
        small_limit = bnb_cfg.get("small_instance_max_customers")
        if small_limit and len(instance.customer_ids) <= small_limit:
            # small-instance mode: deeper LDS budget, still timeout-capped
            max_discrepancy = int(bnb_cfg.get("small_max_discrepancy", 3))
            max_nodes = int(bnb_cfg.get("small_max_nodes", max_nodes))
        result = run_cvrp_bnb_lds(instance, max_discrepancy=max_discrepancy,
                                  max_nodes=max_nodes, seed=seed,
                                  timeout_sec=timeout_sec)
        budget_name = "max_nodes"
    else:
        raise ValueError(f"unknown CVRP algorithm '{algorithm}'")

    solution_path = None
    if output_dir is not None:
        solution_path = _write_solution_file(
            instance, result.best_solution, output_dir, algorithm, seed,
            include_unused_vehicles,
        )
    row = cvrp_result_to_row(result, instance_path, seed, budget_name, budget,
                             timeout_sec, solution_path=solution_path,
                             bks_cost=bks_cost)
    if algorithm == "alns_enhanced":
        # both ALNS variants return algorithm="cvrp_alns"; keep the rows apart
        row["algorithm"] = "cvrp_alns_enhanced"
    return row


def _error_row(instance_name, instance_path, algorithm, seed, timeout_sec, message) -> dict:
    row = {field: "" for field in CVRP_FIELDNAMES}
    row.update({
        "part": "cvrp",
        "instance": instance_name,
        "instance_path": str(instance_path),
        "algorithm": algorithm,
        "seed": seed,
        "timeout_sec": timeout_sec,
        "feasible": False,
        "errors": message,
    })
    return row


def run_cvrp_experiments(instance_paths, algorithms, seeds, budget=100,
                         timeout_sec=10.0, output_dir=None,
                         include_unused_vehicles=False,
                         bks_by_instance=None, tuned=None) -> list[dict]:
    rows = []
    for instance_path in instance_paths:
        instance = parse_cvrplib(instance_path)
        bks_cost = (bks_by_instance or {}).get(instance.name)
        for algorithm in algorithms:
            for seed in seeds:
                try:
                    rows.append(run_one_cvrp_algorithm(
                        instance, instance_path, algorithm, seed, budget,
                        timeout_sec, output_dir=output_dir,
                        include_unused_vehicles=include_unused_vehicles,
                        bks_cost=bks_cost, tuned=tuned,
                    ))
                except Exception as exc:
                    # one failing run should not kill the whole experiment
                    rows.append(_error_row(instance.name, instance_path,
                                           algorithm, seed, timeout_sec, str(exc)))
    return rows
