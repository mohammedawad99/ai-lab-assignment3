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


def load_tuned():
    import json
    path = REPO_ROOT / "configs" / "tuned_cvrp_settings.json"
    return json.loads(path.read_text()) if path.exists() else {}


TUNED = load_tuned()


def run_case(instance, algorithm, budget, timeout_sec):
    # matches the final settings, including the Stage 11-B advanced pass
    # (size-gated by customer count); the plotted instances (P, A80, X) all
    # use the enhanced ALNS variant under the pre-declared policy
    if algorithm == "alns":
        alns_cfg = TUNED.get("alns", {})
        adv_cfg = TUNED.get("alns_advanced", {})
        advanced = bool(adv_cfg.get("advanced_local_search", False)) and \
            len(instance.customer_ids) >= int(adv_cfg.get("min_customers", 0))
        return run_cvrp_alns(instance, iterations=budget, seed=SEED,
                             timeout_sec=timeout_sec,
                             enhanced_operators=True,
                             reaction_rate=alns_cfg.get("reaction_rate", 0.2),
                             advanced_local_search=advanced,
                             advanced_every=int(adv_cfg.get("advanced_every", 25)),
                             advanced_max_passes=int(adv_cfg.get("advanced_max_passes", 2)),
                             candidate_list_k=adv_cfg.get("candidate_list_k"))
    if algorithm == "aco":
        return run_cvrp_aco(instance, iterations=budget, ants=10, seed=SEED,
                            timeout_sec=timeout_sec)
    if algorithm == "tabu":
        return run_cvrp_tabu_search(instance, iterations=budget, seed=SEED,
                                    timeout_sec=timeout_sec)
    if algorithm == "ga_island":
        ga_cfg = TUNED.get("ga_island", {})
        ga_advanced = bool(ga_cfg.get("advanced_local_search", False)) and \
            len(instance.customer_ids) >= int(ga_cfg.get("advanced_min_customers", 0))
        return run_cvrp_ga_island(
            instance, generations=budget,
            population_size=int(ga_cfg.get("population_size", 12)),
            islands=int(ga_cfg.get("islands", 2)),
            mutation_rate=ga_cfg.get("mutation_rate", 0.15),
            seed=SEED, timeout_sec=timeout_sec,
            advanced_local_search=ga_advanced,
            local_search_every=int(ga_cfg.get("local_search_every", 10)),
            advanced_max_passes=int(ga_cfg.get("advanced_max_passes", 1)),
            candidate_list_k=ga_cfg.get("candidate_list_k"))
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
        # plot every recorded operator weight (enhanced mode has more of them)
        weight_keys = sorted(k for k in history[0] if k.endswith("_weight"))
        if weight_keys:
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
