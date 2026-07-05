"""Stage 11-A performance profiling for the CVRP algorithms.

Runs a small, fixed experiment grid (instances x algorithms x seeds) with a
short time limit and records runtime, feasibility and gap per run, so the
local-search optimizations can be compared before/after on identical settings.

With --profile-local-search it additionally cProfiles one representative run
per instance and reports the cumulative time spent inside the local-search
functions (2-opt, relocate, route_cost), which is where the O(L^3) full
recomputation showed up.

Outputs are local-only, under results/performance/ (never committed):
    results/performance/stage11a_<tag>_profile.csv
    results/performance/stage11a_<tag>_summary.txt

Usage:
    .venv/bin/python scripts/profile_cvrp_performance.py \
        --tag baseline --instances P-n16-k8 A-n32-k5 A-n80-k10 \
        --algorithms baseline alns --seeds 42 43 --time-limit 4
"""

import argparse
import cProfile
import csv
import io
import json
import pstats
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.cvrp.io_cvrplib import parse_cvrplib
from src.experiments.cvrp_experiments import run_one_cvrp_algorithm

DATA_DIR = REPO_ROOT / "data" / "official_cvrp"
OUTPUT_DIR = REPO_ROOT / "results" / "performance"
BKS_PATH = REPO_ROOT / "data" / "cvrp_bks.csv"
TUNED_PATH = REPO_ROOT / "configs" / "tuned_cvrp_settings.json"

DEFAULT_INSTANCES = ["P-n16-k8", "A-n32-k5", "A-n80-k10", "X-n101-k25", "M-n200-k17"]
DEFAULT_ALGORITHMS = ["baseline", "alns", "ga_island", "sa", "tabu"]

# local-search functions whose cumulative time we extract from cProfile
HOTSPOT_FUNCTIONS = [
    "improve_route_2opt",
    "improve_route_2opt_full",
    "two_opt_route",
    "two_opt_route_delta",
    "improve_solution_2opt",
    "relocate_best_improvement_pass",
    "route_cost",
    "solution_cost",
]


def load_bks() -> dict:
    with open(BKS_PATH, newline="") as f:
        return {row["instance"]: float(row["bks_cost"]) for row in csv.DictReader(f)}


def load_tuned() -> dict:
    if TUNED_PATH.exists():
        return json.loads(TUNED_PATH.read_text())
    return {}


def profile_one(instance, instance_path, algorithm, seed, budget, timeout, tuned):
    """cProfile one run and return {function_name: cumulative_seconds}."""
    profiler = cProfile.Profile()
    profiler.enable()
    run_one_cvrp_algorithm(instance, instance_path, algorithm, seed, budget,
                           timeout, bks_cost=None, tuned=tuned)
    profiler.disable()
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    hotspots = {}
    for (filename, _line, name), stat in stats.stats.items():
        if name in HOTSPOT_FUNCTIONS and "cvrp" in filename:
            # stat = (call_count, ncalls, tottime, cumtime, callers)
            hotspots[name] = {"calls": stat[0], "cumtime": stat[3]}
    return hotspots


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", default="baseline",
                        help="output name part, e.g. baseline or optimized")
    parser.add_argument("--instances", nargs="+", default=DEFAULT_INSTANCES)
    parser.add_argument("--algorithms", nargs="+", default=DEFAULT_ALGORITHMS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
    parser.add_argument("--budget", type=int, default=200)
    parser.add_argument("--time-limit", type=float, default=4.0)
    parser.add_argument("--profile-local-search", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bks = load_bks()
    tuned = load_tuned()

    rows = []
    for name in args.instances:
        path = DATA_DIR / f"{name}.vrp"
        if not path.exists():
            print(f"skip {name}: no official data file")
            continue
        instance = parse_cvrplib(path)
        for algorithm in args.algorithms:
            for seed in args.seeds:
                start = time.perf_counter()
                row = run_one_cvrp_algorithm(
                    instance, path, algorithm, seed, args.budget,
                    args.time_limit, bks_cost=bks.get(name), tuned=tuned)
                wall = time.perf_counter() - start
                rows.append({
                    "instance": name,
                    "algorithm": row["algorithm"],
                    "seed": seed,
                    "wall_time_sec": f"{wall:.4f}",
                    "solver_time_sec": row["elapsed_time"],
                    "best_cost": row["best_cost"],
                    "gap_percent": row["gap_percent"],
                    "feasible": row["feasible"],
                    "iterations": row["iterations"],
                })
                print(f"{name:12s} {row['algorithm']:22s} seed {seed}: "
                      f"{wall:7.2f}s  gap {row['gap_percent']}%  "
                      f"iters {row['iterations']}")

    csv_path = OUTPUT_DIR / f"stage11a_{args.tag}_profile.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [f"Stage 11-A profile ({args.tag})",
             f"grid: {args.instances} x {args.algorithms} x seeds {args.seeds}",
             f"budget {args.budget}, time limit {args.time_limit}s", ""]
    by_algo = {}
    for row in rows:
        by_algo.setdefault(row["algorithm"], []).append(row)
    for algorithm, algo_rows in sorted(by_algo.items()):
        total = sum(float(r["wall_time_sec"]) for r in algo_rows)
        gaps = [float(r["gap_percent"]) for r in algo_rows if r["gap_percent"]]
        feasible = all(str(r["feasible"]).lower() == "true" for r in algo_rows)
        lines.append(f"{algorithm:22s} total {total:8.2f}s  "
                     f"mean gap {sum(gaps) / len(gaps):7.3f}%  "
                     f"feasible {'yes' if feasible else 'NO'}")

    if args.profile_local_search:
        lines += ["", "local-search hotspots (cProfile, one alns run per instance):"]
        for name in args.instances:
            path = DATA_DIR / f"{name}.vrp"
            if not path.exists():
                continue
            instance = parse_cvrplib(path)
            hotspots = profile_one(instance, path, "alns", args.seeds[0],
                                   args.budget, args.time_limit, tuned)
            lines.append(f"  {name}:")
            for func, stat in sorted(hotspots.items(),
                                     key=lambda kv: -kv[1]["cumtime"]):
                lines.append(f"    {func:32s} {stat['calls']:>10d} calls  "
                             f"{stat['cumtime']:8.3f}s cumulative")

    summary_path = OUTPUT_DIR / f"stage11a_{args.tag}_summary.txt"
    summary_path.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {csv_path}\nwrote {summary_path}")


if __name__ == "__main__":
    main()
