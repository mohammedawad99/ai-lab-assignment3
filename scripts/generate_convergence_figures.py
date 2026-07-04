"""Generate convergence plots for selected CVRP cases.

The final raw CSVs store only end results, so this script reruns a small,
fixed set of (instance, algorithm) cases with seed 42 and the planned
budgets/timeouts, and plots the best-cost history that each solver records
in its convergence list. The runs are deterministic, so the end costs match
the final result rows. This is NOT a rerun of the full suite.

Usage:
    python scripts/generate_convergence_figures.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.cvrp.io_cvrplib import parse_cvrplib
from src.cvrp.solvers.aco import run_cvrp_aco
from src.cvrp.solvers.alns import run_cvrp_alns
from src.cvrp.solvers.ga_island import run_cvrp_ga_island
from src.cvrp.solvers.tabu_search import run_cvrp_tabu_search
from src.experiments.final_plan import load_final_plan

FIGURES = REPO_ROOT / "report" / "figures"
DATA_DIR = REPO_ROOT / "data" / "official_cvrp"
SEED = 42

# (instance, [algorithms]) — small fixed visualization set from the prompt
CASES = [
    ("P-n16-k8", ["alns", "aco", "tabu"]),
    ("A-n80-k10", ["alns", "ga_island", "aco"]),
    ("X-n101-k25", ["alns", "aco"]),
]


def run_case(instance, algorithm, budget, timeout_sec):
    if algorithm == "alns":
        return run_cvrp_alns(instance, iterations=budget, seed=SEED,
                             timeout_sec=timeout_sec)
    if algorithm == "aco":
        return run_cvrp_aco(instance, iterations=budget, ants=10, seed=SEED,
                            timeout_sec=timeout_sec)
    if algorithm == "tabu":
        return run_cvrp_tabu_search(instance, iterations=budget, seed=SEED,
                                    timeout_sec=timeout_sec)
    if algorithm == "ga_island":
        return run_cvrp_ga_island(instance, generations=budget,
                                  population_size=12, islands=2, seed=SEED,
                                  timeout_sec=timeout_sec)
    raise SystemExit(f"unknown algorithm '{algorithm}'")


def main():
    plan = load_final_plan(REPO_ROOT / "configs" / "final_experiment_plan.json")
    per_instance = plan["cvrp"]["per_instance"]
    FIGURES.mkdir(parents=True, exist_ok=True)
    alns_weights_result = None

    for instance_name, algorithms in CASES:
        instance_path = DATA_DIR / f"{instance_name}.vrp"
        if not instance_path.exists():
            raise SystemExit(f"missing official instance: {instance_path}")
        instance = parse_cvrplib(instance_path)
        settings = per_instance[instance_name]

        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        for algorithm in algorithms:
            result = run_case(instance, algorithm, settings["budget"],
                              settings["timeout_sec"])
            history = result.convergence
            if not history or "best_cost" not in history[0]:
                print(f"note: {algorithm} on {instance_name} exposes no "
                      "usable convergence history, skipped in plot")
                continue
            xs = [row["iteration"] for row in history]
            ys = [row["best_cost"] for row in history]
            ax.plot(xs, ys, marker=".", markersize=3, label=algorithm)
            print(f"ran {instance_name} {algorithm}: "
                  f"final best {result.best_cost:.2f}, "
                  f"{len(history)} history points")
            if instance_name == "A-n80-k10" and algorithm == "alns":
                alns_weights_result = result

        ax.set_title(f"CVRP convergence — {instance_name} "
                     f"(seed {SEED}, budget {settings['budget']})")
        ax.set_xlabel("iteration / generation")
        ax.set_ylabel("best cost")
        ax.legend(fontsize=8)
        fig.tight_layout()
        output = FIGURES / f"convergence_{instance_name}.png"
        fig.savefig(output, dpi=150)
        plt.close(fig)
        print(f"generated: {output.relative_to(REPO_ROOT)}")

    # ALNS operator weights on A-n80-k10 (recorded in its convergence rows)
    if alns_weights_result is not None:
        history = alns_weights_result.convergence
        weight_keys = ["random_removal_weight", "worst_removal_weight",
                       "greedy_repair_weight", "regret2_repair_weight"]
        if all(key in history[0] for key in weight_keys):
            fig, ax = plt.subplots(figsize=(7.2, 4.0))
            xs = [row["iteration"] for row in history]
            for key in weight_keys:
                ax.plot(xs, [row[key] for row in history], marker=".",
                        markersize=3, label=key.replace("_weight", ""))
            ax.set_title("ALNS adaptive operator weights — A-n80-k10 (seed 42)")
            ax.set_xlabel("iteration")
            ax.set_ylabel("weight")
            ax.legend(fontsize=8)
            fig.tight_layout()
            output = FIGURES / "alns_operator_weights_A-n80-k10.png"
            fig.savefig(output, dpi=150)
            plt.close(fig)
            print(f"generated: {output.relative_to(REPO_ROOT)}")
        else:
            print("note: ALNS convergence rows do not include operator "
                  "weights, weights figure skipped")
    sys.exit(0)


if __name__ == "__main__":
    main()
