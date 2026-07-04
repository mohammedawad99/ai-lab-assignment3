"""Controlled tuning grids for the CVRP solvers (ALNS, GA-Island, SA).

Runs compact, named configurations over selected official instances and
seeds, using the per-instance budgets/timeouts from the final plan. Results
are appended to a CSV under results/tuning/ (resumable: existing
config/instance/seed rows are skipped) and a mean-gap leaderboard is
printed. Nothing here changes any solver default.

Usage:
    python scripts/tune_cvrp_algorithms.py --algorithms alns ga_island sa \
        --instances P-n16-k8 A-n32-k5 A-n80-k10 --seeds 42 43 44
    python scripts/tune_cvrp_algorithms.py --validate alns_enh alns_base \
        --instances P-n16-k8 E-n22-k4 A-n32-k5 A-n80-k10 X-n101-k25 M-n200-k17 \
        --output results/tuning/stage10a_validation_results.csv
"""

import argparse
import csv
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.cvrp.io_cvrplib import parse_cvrplib
from src.cvrp.solvers.alns import run_cvrp_alns
from src.cvrp.solvers.ga_island import run_cvrp_ga_island
from src.cvrp.solvers.simulated_annealing import run_cvrp_simulated_annealing
from src.experiments.final_plan import load_final_plan
from src.experiments.official_benchmarks import expected_instance_path, load_bks_table

FIELDNAMES = ["config", "algorithm", "instance", "seed", "best_cost",
              "gap_percent", "feasible", "elapsed_time", "iterations"]

# ---- compact named configuration grids ----
# ALNS: kwargs passed on top of (iterations=budget, seed, timeout)
ALNS_CONFIGS = {
    "alns_base": {},
    "alns_base_rm40": {"max_removal_fraction": 0.4},
    "alns_enh": {"enhanced_operators": True},
    "alns_enh_rm40": {"enhanced_operators": True, "max_removal_fraction": 0.4},
    "alns_enh_cool999": {"enhanced_operators": True, "cooling_rate": 0.999},
    "alns_enh_t20": {"enhanced_operators": True, "initial_temperature": 20.0},
    "alns_enh_react50": {"enhanced_operators": True, "reaction_rate": 0.5},
    "alns_enh_rm40_cool999": {"enhanced_operators": True,
                              "max_removal_fraction": 0.4, "cooling_rate": 0.999},
}

# GA-Island: kwargs on top of (generations=budget, seed, timeout)
GA_CONFIGS = {
    "ga_base": {"population_size": 12, "islands": 2},
    "ga_pop30": {"population_size": 30, "islands": 2},
    "ga_pop30_isl4": {"population_size": 30, "islands": 4},
    "ga_pop30_mut30": {"population_size": 30, "islands": 2, "mutation_rate": 0.3},
    "ga_pop50_isl4": {"population_size": 50, "islands": 4},
    "ga_pop30_mig10": {"population_size": 30, "islands": 2, "migration_interval": 10},
}

# SA: "iters_factor" multiplies the plan budget (SA iterations are very
# cheap, so the same time budget allows far more of them)
SA_CONFIGS = {
    "sa_base": {},
    "sa_it50": {"iters_factor": 50, "cooling_rate": 0.9995},
    "sa_it50_t20": {"iters_factor": 50, "cooling_rate": 0.9995,
                    "initial_temperature": 20.0},
    "sa_it200": {"iters_factor": 200, "cooling_rate": 0.9999},
    "sa_it200_t500": {"iters_factor": 200, "cooling_rate": 0.9999,
                      "initial_temperature": 500.0},
    "sa_it50_t500": {"iters_factor": 50, "cooling_rate": 0.9995,
                     "initial_temperature": 500.0},
}

ALL_CONFIGS = {}
for name, kwargs in ALNS_CONFIGS.items():
    ALL_CONFIGS[name] = ("alns", kwargs)
for name, kwargs in GA_CONFIGS.items():
    ALL_CONFIGS[name] = ("ga_island", kwargs)
for name, kwargs in SA_CONFIGS.items():
    ALL_CONFIGS[name] = ("sa", kwargs)


def run_config(algorithm, kwargs, instance, budget, timeout_sec, seed):
    if algorithm == "alns":
        return run_cvrp_alns(instance, iterations=budget, seed=seed,
                             timeout_sec=timeout_sec, **kwargs)
    if algorithm == "ga_island":
        return run_cvrp_ga_island(instance, generations=budget, seed=seed,
                                  timeout_sec=timeout_sec, **kwargs)
    if algorithm == "sa":
        kwargs = dict(kwargs)
        factor = kwargs.pop("iters_factor", 1)
        return run_cvrp_simulated_annealing(instance, iterations=budget * factor,
                                            seed=seed, timeout_sec=timeout_sec,
                                            **kwargs)
    raise SystemExit(f"unknown algorithm '{algorithm}'")


def existing_keys(path):
    if not path.exists():
        return set()
    with open(path, newline="") as f:
        return {(r["config"], r["instance"], r["seed"])
                for r in csv.DictReader(f)}


def main():
    parser = argparse.ArgumentParser(description="Tune CVRP solvers on a compact grid.")
    parser.add_argument("--algorithms", nargs="+", default=["alns"],
                        choices=["alns", "ga_island", "sa"])
    parser.add_argument("--validate", nargs="+",
                        help="run only these named configs (skips --algorithms)")
    parser.add_argument("--instances", nargs="+",
                        default=["P-n16-k8", "A-n32-k5", "A-n80-k10"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--output", default="results/tuning/stage10a_tuning_results.csv")
    args = parser.parse_args()

    plan = load_final_plan(REPO_ROOT / "configs" / "final_experiment_plan.json")
    per_instance = plan["cvrp"]["per_instance"]
    bks = load_bks_table(REPO_ROOT / "data" / "cvrp_bks.csv")

    if args.validate:
        configs = {}
        for name in args.validate:
            if name not in ALL_CONFIGS:
                sys.exit(f"unknown config '{name}'")
            configs[name] = ALL_CONFIGS[name]
    else:
        configs = {name: spec for name, spec in ALL_CONFIGS.items()
                   if spec[0] in args.algorithms}

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    done = existing_keys(output)
    is_new = not output.exists()

    with open(output, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if is_new:
            writer.writeheader()
        for instance_name in args.instances:
            instance = parse_cvrplib(expected_instance_path(instance_name))
            settings = per_instance[instance_name]
            for config_name, (algorithm, kwargs) in configs.items():
                for seed in args.seeds:
                    if (config_name, instance_name, str(seed)) in done:
                        continue
                    start = time.perf_counter()
                    result = run_config(algorithm, kwargs, instance,
                                        settings["budget"],
                                        settings["timeout_sec"], seed)
                    gap = 100.0 * (result.best_cost - bks[instance_name]) / bks[instance_name]
                    writer.writerow({
                        "config": config_name, "algorithm": algorithm,
                        "instance": instance_name, "seed": seed,
                        "best_cost": f"{result.best_cost:.4f}",
                        "gap_percent": f"{gap:.4f}",
                        "feasible": result.feasible,
                        "elapsed_time": f"{time.perf_counter() - start:.3f}",
                        "iterations": result.iterations,
                    })
                    f.flush()
                    print(f"{config_name} {instance_name} seed {seed}: "
                          f"cost {result.best_cost:.2f} gap {gap:.2f}% "
                          f"feasible {result.feasible}")

    # leaderboard: mean gap per config over everything in the CSV
    with open(output, newline="") as f:
        rows = [r for r in csv.DictReader(f)
                if r["config"] in configs and r["instance"] in args.instances]
    print("\n== leaderboard (mean gap over instances/seeds) ==")
    board = {}
    for row in rows:
        board.setdefault(row["config"], []).append(float(row["gap_percent"]))
    for name, gaps in sorted(board.items(), key=lambda kv: sum(kv[1]) / len(kv[1])):
        print(f"{name}: mean gap {sum(gaps) / len(gaps):.3f}% over {len(gaps)} runs")
    sys.exit(0)


if __name__ == "__main__":
    main()
