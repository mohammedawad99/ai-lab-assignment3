"""Print the report facts from the final result CSVs (and optionally save JSON).

This only reads results and prints them — it never edits the report, so the
numbers in report/assignment3_report.md can be checked against this output.

Usage:
    python scripts/extract_report_numbers.py \
        [--results-dir results/final_experiments] \
        [--output results/final_experiments/report_numbers.json]
"""

import argparse
import csv
import json
import sys
from pathlib import Path

INSTANCES = ["P-n16-k8", "E-n22-k4", "A-n32-k5", "A-n80-k10", "X-n101-k25", "M-n200-k17"]


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description="Extract report numbers.")
    parser.add_argument("--results-dir", default="results/final_experiments")
    parser.add_argument("--output", help="optional JSON snapshot path")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    facts = {}

    # ---- CVRP ----
    cvrp_rows = read_rows(results_dir / "raw" / "cvrp_all_instances.csv")
    feasible = [r for r in cvrp_rows if r["feasible"] == "True"]
    with_errors = [r for r in cvrp_rows if r["errors"]]
    facts["cvrp_rows"] = len(cvrp_rows)
    facts["cvrp_feasible_rows"] = len(feasible)
    facts["cvrp_error_rows"] = len(with_errors)
    print(f"cvrp rows: {len(cvrp_rows)}, feasible: {len(feasible)}, "
          f"with errors: {len(with_errors)}")

    facts["cvrp_best_per_instance"] = {}
    for name in INSTANCES:
        rows = [r for r in feasible if r["instance"] == name and r["gap_percent"]]
        if not rows:
            print(f"  {name}: no feasible rows")
            continue
        best = min(rows, key=lambda r: (float(r["gap_percent"]), float(r["best_cost"])))
        facts["cvrp_best_per_instance"][name] = {
            "algorithm": best["algorithm"],
            "best_cost": float(best["best_cost"]),
            "gap_percent": float(best["gap_percent"]),
            "feasible_runs": len([r for r in feasible if r["instance"] == name]),
        }
        print(f"  best {name}: {best['algorithm']} cost {best['best_cost']} "
              f"gap {best['gap_percent']}%")

    facts["cvrp_mean_gap_per_algorithm"] = {}
    for algo in sorted({r["algorithm"] for r in cvrp_rows}):
        gaps = [float(r["gap_percent"]) for r in feasible
                if r["algorithm"] == algo and r["gap_percent"]]
        if gaps:
            facts["cvrp_mean_gap_per_algorithm"][algo] = sum(gaps) / len(gaps)
            print(f"  mean gap {algo}: {sum(gaps) / len(gaps):.2f}%")

    # ---- Ackley ----
    ackley_rows = read_rows(results_dir / "raw" / "ackley_d10.csv")
    facts["ackley_rows"] = len(ackley_rows)
    facts["ackley_error_rows"] = len([r for r in ackley_rows if r["errors"]])
    best = min(ackley_rows, key=lambda r: float(r["best_value"]))
    facts["ackley_best"] = {"algorithm": best["algorithm"], "seed": best["seed"],
                            "best_value": float(best["best_value"])}
    print(f"ackley rows: {len(ackley_rows)}, errors: {facts['ackley_error_rows']}")
    print(f"  best ackley: {best['algorithm']} seed {best['seed']} "
          f"value {best['best_value']}")

    # ---- GP/GEP ----
    gp_rows = read_rows(results_dir / "raw" / "gp_gep_comparison_runs.csv")
    facts["gp_gep_rows"] = len(gp_rows)
    best = max(gp_rows, key=lambda r: float(r["eval_fitness"]))
    facts["gp_gep_best"] = {"algorithm": best["algorithm"], "seed": best["seed"],
                            "eval_fitness": float(best["eval_fitness"]),
                            "eval_solved": best["eval_solved_count"]}
    expressions = [r["best_expression"] for r in gp_rows]
    facts["gp_gep_expression_diversity"] = len(set(expressions)) / len(expressions)
    print(f"gp/gep rows: {len(gp_rows)}")
    print(f"  best: {best['algorithm']} seed {best['seed']} "
          f"eval_fitness {best['eval_fitness']} solved {best['eval_solved_count']}")
    print(f"  expression diversity over all runs: "
          f"{facts['gp_gep_expression_diversity']:.2f}")
    summary_txt = results_dir / "raw" / "gp_gep_comparison_summary.txt"
    if summary_txt.exists():
        print("  summary txt:")
        for line in summary_txt.read_text().splitlines():
            print(f"    {line}")

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(facts, f, indent=2)
        print(f"json written to: {output}")
    sys.exit(0)


if __name__ == "__main__":
    main()
