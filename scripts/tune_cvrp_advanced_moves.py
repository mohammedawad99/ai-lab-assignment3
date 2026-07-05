"""Stage 11-B validation: advanced local-search moves vs current tuned setup.

Runs ALNS and GA-Island with the current settings and with the new advanced
intensification (relocate + swap + Or-opt + 2-opt*) under several
candidate-list sizes, on scaled-down budgets that mirror the final plan's
per-instance proportions. Everything is local-only under results/stage11b/.

Usage:
    .venv/bin/python scripts/tune_cvrp_advanced_moves.py \
        [--instances ...] [--seeds 42 43 44] [--algorithms alns ga_island]
"""

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.cvrp.io_cvrplib import parse_cvrplib
from src.cvrp.solvers.alns import run_cvrp_alns
from src.cvrp.solvers.ga_island import run_cvrp_ga_island

DATA_DIR = REPO_ROOT / "data" / "official_cvrp"
OUTPUT_DIR = REPO_ROOT / "results" / "stage11b"
BKS_PATH = REPO_ROOT / "data" / "cvrp_bks.csv"
TUNED_PATH = REPO_ROOT / "configs" / "tuned_cvrp_settings.json"

DEFAULT_INSTANCES = ["P-n16-k8", "A-n32-k5", "A-n80-k10",
                     "X-n101-k25", "M-n200-k17"]

# scaled-down proxies of the final plan (same ordering of effort by size)
VALIDATION_BUDGETS = {
    "P-n16-k8": (300, 10.0),
    "E-n22-k4": (300, 10.0),
    "A-n32-k5": (300, 20.0),
    "A-n80-k10": (400, 40.0),
    "X-n101-k25": (500, 50.0),
    "M-n200-k17": (500, 70.0),
}

ALNS_SETTINGS = [
    {"name": "current", "advanced": False},
    {"name": "adv-full", "advanced": True, "k": None, "every": 25, "passes": 2},
    {"name": "adv-k10", "advanced": True, "k": 10, "every": 25, "passes": 2},
    {"name": "adv-k20", "advanced": True, "k": 20, "every": 25, "passes": 2},
]

GA_SETTINGS = [
    {"name": "current", "advanced": False},
    {"name": "adv-k10", "advanced": True, "k": 10, "every": 10, "passes": 1},
    {"name": "adv-k20", "advanced": True, "k": 20, "every": 10, "passes": 1},
]


def load_bks():
    with open(BKS_PATH, newline="") as f:
        return {row["instance"]: float(row["bks_cost"]) for row in csv.DictReader(f)}


def run_alns(instance, name, setting, seed, tuned):
    budget, timeout = VALIDATION_BUDGETS[name]
    # final policy: enhanced operators everywhere except the excluded instances
    excluded = tuned.get("alns_policy", {}).get("enhanced_excluded_instances", [])
    enhanced = name not in excluded
    kwargs = {}
    if enhanced:
        kwargs["enhanced_operators"] = True
        kwargs["reaction_rate"] = tuned.get("alns", {}).get("reaction_rate", 0.2)
    if setting["advanced"]:
        kwargs.update(advanced_local_search=True,
                      candidate_list_k=setting["k"],
                      advanced_every=setting["every"],
                      advanced_max_passes=setting["passes"])
    return run_cvrp_alns(instance, iterations=budget, seed=seed,
                         timeout_sec=timeout, **kwargs)


def run_ga(instance, name, setting, seed, tuned):
    budget, timeout = VALIDATION_BUDGETS[name]
    ga_cfg = tuned.get("ga_island", {})
    kwargs = dict(population_size=int(ga_cfg.get("population_size", 12)),
                  islands=int(ga_cfg.get("islands", 2)),
                  mutation_rate=ga_cfg.get("mutation_rate", 0.15))
    if setting["advanced"]:
        kwargs.update(advanced_local_search=True,
                      candidate_list_k=setting["k"],
                      local_search_every=setting["every"],
                      advanced_max_passes=setting["passes"])
    return run_cvrp_ga_island(instance, generations=budget, seed=seed,
                              timeout_sec=timeout, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--instances", nargs="+", default=DEFAULT_INSTANCES)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--algorithms", nargs="+", default=["alns", "ga_island"])
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bks = load_bks()
    tuned = json.loads(TUNED_PATH.read_text()) if TUNED_PATH.exists() else {}

    grids = {"alns": (ALNS_SETTINGS, run_alns), "ga_island": (GA_SETTINGS, run_ga)}
    rows = []
    for name in args.instances:
        path = DATA_DIR / f"{name}.vrp"
        if not path.exists():
            print(f"skip {name}: no official data file")
            continue
        instance = parse_cvrplib(path)
        for algorithm in args.algorithms:
            settings, runner = grids[algorithm]
            for setting in settings:
                for seed in args.seeds:
                    started = time.perf_counter()
                    result = runner(instance, name, setting, seed, tuned)
                    wall = time.perf_counter() - started
                    gap = 100.0 * (result.best_cost - bks[name]) / bks[name]
                    rows.append({
                        "instance": name, "algorithm": algorithm,
                        "setting": setting["name"], "seed": seed,
                        "best_cost": f"{result.best_cost:.4f}",
                        "gap_percent": f"{gap:.4f}",
                        "feasible": result.feasible,
                        "iterations": result.iterations,
                        "wall_time_sec": f"{wall:.2f}",
                    })
                    print(f"{name:12s} {algorithm:10s} {setting['name']:9s} "
                          f"seed {seed}: gap {gap:7.3f}%  {wall:6.1f}s  "
                          f"iters {result.iterations}  "
                          f"feasible {result.feasible}")

    with open(OUTPUT_DIR / "advanced_moves_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # summary: mean/median gap and mean runtime per instance x algorithm x setting
    summary = []
    keys = sorted({(r["instance"], r["algorithm"], r["setting"]) for r in rows})
    for instance, algorithm, setting in keys:
        group = [r for r in rows if (r["instance"], r["algorithm"],
                                     r["setting"]) == (instance, algorithm, setting)]
        gaps = [float(r["gap_percent"]) for r in group]
        summary.append({
            "instance": instance, "algorithm": algorithm, "setting": setting,
            "mean_gap": f"{statistics.mean(gaps):.4f}",
            "median_gap": f"{statistics.median(gaps):.4f}",
            "best_gap": f"{min(gaps):.4f}",
            "mean_time_sec": f"{statistics.mean(float(r['wall_time_sec']) for r in group):.2f}",
            "all_feasible": all(str(r["feasible"]).lower() == "true" for r in group),
        })
    with open(OUTPUT_DIR / "advanced_moves_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    lines = ["Stage 11-B advanced-move validation summary", ""]
    for algorithm in args.algorithms:
        lines.append(f"[{algorithm}] mean gap % by setting (rows) and instance:")
        settings = [s["name"] for s in grids[algorithm][0]]
        lines.append(f"  {'setting':10s}" + "".join(f"{i:>13s}" for i in args.instances))
        for setting in settings:
            cells = []
            for instance in args.instances:
                match = [s for s in summary
                         if (s["instance"], s["algorithm"], s["setting"])
                         == (instance, algorithm, setting)]
                cells.append(f"{float(match[0]['mean_gap']):13.4f}" if match
                             else f"{'-':>13s}")
            lines.append(f"  {setting:10s}" + "".join(cells))
        lines.append("")
    (OUTPUT_DIR / "advanced_moves_recommendation.txt").write_text(
        "\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {OUTPUT_DIR}/advanced_moves_results.csv, "
          f"advanced_moves_summary.csv, advanced_moves_recommendation.txt")


if __name__ == "__main__":
    main()
