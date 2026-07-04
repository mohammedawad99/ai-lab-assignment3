"""Seed-robustness runs for the CVRP solvers (extra seeds for boxplots).

Runs the final effective algorithm set (tuned settings, effective ALNS
policy, B&B small-instance mode) over more seeds than the final plan, so
the report can show gap distributions instead of only best values. Results
are appended to a resumable CSV under results/robustness/ (not committed).

Usage:
    python scripts/run_seed_robustness.py
    python scripts/run_seed_robustness.py --instances P-n16-k8 A-n32-k5 --seeds 42 43 44 45 46
"""

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.cvrp.io_cvrplib import parse_cvrplib
from src.experiments.cvrp_experiments import run_one_cvrp_algorithm
from src.experiments.final_plan import load_final_plan
from src.experiments.official_benchmarks import expected_instance_path, load_bks_table

FIELDNAMES = ["instance", "algorithm", "seed", "best_cost", "gap_percent",
              "feasible", "errors", "elapsed_time"]

DEFAULT_INSTANCES = ["P-n16-k8", "A-n32-k5", "A-n80-k10", "X-n101-k25"]
DEFAULT_ALGORITHMS = ["baseline", "sa", "tabu", "aco", "ga_island",
                      "alns_effective", "bnb_lds"]
DEFAULT_SEEDS = [42, 43, 44, 45, 46, 47, 48, 49]


def existing_keys(path):
    if not path.exists():
        return set()
    with open(path, newline="") as f:
        return {(r["instance"], r["algorithm"], r["seed"])
                for r in csv.DictReader(f)}


def main():
    parser = argparse.ArgumentParser(description="CVRP seed-robustness runs.")
    parser.add_argument("--instances", nargs="+", default=DEFAULT_INSTANCES)
    parser.add_argument("--algorithms", nargs="+", default=DEFAULT_ALGORITHMS)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--output",
                        default="results/robustness/cvrp_seed_robustness_raw.csv")
    parser.add_argument("--summary",
                        default="results/robustness/cvrp_seed_robustness_summary.csv")
    args = parser.parse_args()

    plan = load_final_plan(REPO_ROOT / "configs" / "final_experiment_plan.json")
    per_instance = plan["cvrp"]["per_instance"]
    bks = load_bks_table(REPO_ROOT / "data" / "cvrp_bks.csv")
    tuned = json.loads((REPO_ROOT / "configs" / "tuned_cvrp_settings.json").read_text())
    excluded = set(tuned.get("alns_policy", {}).get("enhanced_excluded_instances", []))

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
            for algorithm in args.algorithms:
                # "alns_effective" applies the pre-declared policy per instance
                actual = algorithm
                if algorithm == "alns_effective":
                    actual = "alns" if instance_name in excluded else "alns_enhanced"
                for seed in args.seeds:
                    if (instance_name, algorithm, str(seed)) in done:
                        continue
                    start = time.perf_counter()
                    row = run_one_cvrp_algorithm(
                        instance, str(expected_instance_path(instance_name)),
                        actual, seed, settings["budget"],
                        settings["timeout_sec"], tuned=tuned,
                        bks_cost=bks[instance_name])
                    writer.writerow({
                        "instance": instance_name, "algorithm": algorithm,
                        "seed": seed, "best_cost": row["best_cost"],
                        "gap_percent": row["gap_percent"],
                        "feasible": row["feasible"], "errors": row["errors"],
                        "elapsed_time": f"{time.perf_counter() - start:.3f}",
                    })
                    f.flush()
                    print(f"{instance_name} {algorithm} seed {seed}: "
                          f"gap {row['gap_percent']}% feasible {row['feasible']}")

    # summary: per instance/algorithm gap statistics over seeds
    with open(output, newline="") as f:
        rows = [r for r in csv.DictReader(f)
                if r["instance"] in args.instances and r["algorithm"] in args.algorithms]
    groups = defaultdict(list)
    for r in rows:
        if r["gap_percent"]:
            groups[(r["instance"], r["algorithm"])].append(float(r["gap_percent"]))

    def median(values):
        v = sorted(values)
        n = len(v)
        return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2

    summary_path = Path(args.summary)
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["instance", "algorithm", "seeds", "min_gap",
                         "median_gap", "max_gap", "mean_gap"])
        for (inst, algo), gaps in sorted(groups.items()):
            writer.writerow([inst, algo, len(gaps), f"{min(gaps):.4f}",
                             f"{median(gaps):.4f}", f"{max(gaps):.4f}",
                             f"{sum(gaps)/len(gaps):.4f}"])
    print(f"\nsummary -> {summary_path}")
    for (inst, algo), gaps in sorted(groups.items()):
        print(f"  {inst} {algo}: min {min(gaps):.2f} med {median(gaps):.2f} "
              f"max {max(gaps):.2f} over {len(gaps)} seeds")
    sys.exit(0)


if __name__ == "__main__":
    main()
