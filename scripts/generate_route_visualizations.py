"""Plot the best final CVRP routes for three official instances.

For each of P-n16-k8, A-n80-k10 and X-n101-k25, the script finds the best
feasible (algorithm, seed) row in the final results, loads the matching
solution file that the final runner wrote, and draws the routes on the real
instance coordinates. Nothing is recomputed or invented.

Usage:
    python scripts/generate_route_visualizations.py
"""

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.cvrp.io_cvrplib import parse_cvrplib
from src.cvrp.solution_io import parse_solution_file
from src.experiments.official_benchmarks import load_bks_table

RESULTS = REPO_ROOT / "results" / "final_experiments"
FIGURES = REPO_ROOT / "report" / "figures"
DATA_DIR = REPO_ROOT / "data" / "official_cvrp"

TARGET_INSTANCES = ["P-n16-k8", "A-n80-k10", "X-n101-k25"]


def best_feasible_row(rows, instance):
    candidates = [r for r in rows
                  if r["instance"] == instance and r["feasible"] == "True"]
    if not candidates:
        raise SystemExit(f"no feasible final rows for {instance}")
    return min(candidates, key=lambda r: float(r["best_cost"]))


def solution_path_for(instance, algorithm, seed):
    short = algorithm.replace("cvrp_", "")
    return (RESULTS / "cvrp_solutions" / instance / "cvrp_solutions"
            / f"{instance}_{short}_seed{seed}.txt")


def plot_routes(instance_name, row, bks):
    instance = parse_cvrplib(DATA_DIR / f"{instance_name}.vrp")
    solution_file = solution_path_for(instance_name, row["algorithm"], row["seed"])
    if not solution_file.exists():
        raise SystemExit(f"solution file missing: {solution_file}")
    solution = parse_solution_file(solution_file)

    dense = instance.node_count > 60
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    for route in solution.routes:
        if len(route) <= 2:
            continue
        xs = [instance.coordinates[node][0] for node in route]
        ys = [instance.coordinates[node][1] for node in route]
        ax.plot(xs, ys, linewidth=0.7 if dense else 1.3,
                marker="o", markersize=2.0 if dense else 4.0)
    depot_x, depot_y = instance.coordinates[instance.depot_id]
    ax.plot(depot_x, depot_y, marker="s", markersize=11, color="black",
            zorder=5, label="depot")
    ax.legend(loc="best", fontsize=8)

    cost = float(row["best_cost"])
    gap = float(row["gap_percent"])
    routes_used = sum(1 for r in solution.routes if len(r) > 2)
    ax.set_title(f"{instance_name} — {row['algorithm'].replace('cvrp_', '')} "
                 f"seed {row['seed']}\ncost {cost:.2f}, gap {gap:.2f}% "
                 f"vs BKS {bks[instance_name]:g}, {routes_used} routes")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    output = FIGURES / f"cvrp_route_{instance_name}.png"
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def main():
    all_rows_path = RESULTS / "raw" / "cvrp_all_instances.csv"
    if not all_rows_path.exists():
        raise SystemExit(f"missing final results: {all_rows_path}")
    with open(all_rows_path, newline="") as f:
        rows = list(csv.DictReader(f))
    bks = load_bks_table(REPO_ROOT / "data" / "cvrp_bks.csv")

    FIGURES.mkdir(parents=True, exist_ok=True)
    for instance_name in TARGET_INSTANCES:
        row = best_feasible_row(rows, instance_name)
        output = plot_routes(instance_name, row, bks)
        print(f"generated: {output.relative_to(REPO_ROOT)} "
              f"({row['algorithm']} seed {row['seed']})")
    sys.exit(0)


if __name__ == "__main__":
    main()
