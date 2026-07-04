"""Compact summary of the Stage 10-C final rerun vs the previous evidence.

Reads report/evidence/ (previous final results) and results/final_experiments/
(new rerun), applies the pre-declared ALNS selection policy from
configs/tuned_cvrp_settings.json, and prints/writes the comparison.

Usage:
    python scripts/extract_final_results_v2.py
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
INSTANCES = ["P-n16-k8", "E-n22-k4", "A-n32-k5", "A-n80-k10", "X-n101-k25", "M-n200-k17"]


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description="Summarize the final rerun.")
    parser.add_argument("--results-dir", default="results/final_experiments")
    parser.add_argument("--output",
                        default="results/final_experiments/final_v2_summary.txt")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    lines = ["Stage 10-C final rerun summary", ""]

    # ---- CVRP ----
    rows = read_rows(results_dir / "raw" / "cvrp_all_instances.csv")
    with open(REPO_ROOT / "configs" / "tuned_cvrp_settings.json") as f:
        policy = json.load(f).get("alns_policy", {})
    excluded = set(policy.get("enhanced_excluded_instances", []))

    def effective_algorithm(row):
        """Apply the pre-declared ALNS rule: the assignment 'alns' result is
        the enhanced variant except on excluded instances."""
        algo, inst = row["algorithm"], row["instance"]
        if algo == "cvrp_alns":
            return "cvrp_alns" if inst in excluded else None  # basic used only there
        if algo == "cvrp_alns_enhanced":
            return None if inst in excluded else "cvrp_alns"
        return algo

    effective = []
    for row in rows:
        name = effective_algorithm(row)
        if name is not None:
            effective.append(dict(row, algorithm=name))

    old = read_rows(REPO_ROOT / "report" / "evidence" / "cvrp_all_summary.csv")

    def old_best(inst):
        return min(float(r["best_gap_percent"]) for r in old if r["instance"] == inst)

    lines.append("== CVRP: best feasible gap per instance (old evidence -> new rerun) ==")
    feasible = [r for r in effective if r["feasible"] == "True" and r["gap_percent"]]
    improved, regressed = [], []
    for inst in INSTANCES:
        new = min(float(r["gap_percent"]) for r in feasible if r["instance"] == inst)
        prev = old_best(inst)
        delta = new - prev
        mark = "IMPROVED" if delta < -0.01 else ("same" if abs(delta) <= 0.01 else "REGRESSED")
        if mark == "IMPROVED":
            improved.append(inst)
        if mark == "REGRESSED":
            regressed.append(inst)
        best_row = min((r for r in feasible if r["instance"] == inst),
                       key=lambda r: float(r["gap_percent"]))
        lines.append(f"  {inst}: {prev:.2f}% -> {new:.2f}% ({delta:+.2f}) {mark} "
                     f"[{best_row['algorithm']}]")
    lines.append(f"  improved: {', '.join(improved) or 'none'}; "
                 f"regressed: {', '.join(regressed) or 'none'}")
    lines.append("")

    lines.append("== CVRP: algorithm mean of best-of-seeds gaps ==")
    baseline_best = {}
    best_by = defaultdict(dict)
    for r in feasible:
        g = float(r["gap_percent"])
        cur = best_by[r["algorithm"]].get(r["instance"])
        if cur is None or g < cur:
            best_by[r["algorithm"]][r["instance"]] = g
    baseline_best = best_by.get("baseline", {})
    for algo in sorted(best_by):
        gaps = [best_by[algo][i] for i in INSTANCES if i in best_by[algo]]
        beats = [i for i in INSTANCES
                 if best_by[algo].get(i, 1e9) < baseline_best.get(i, 1e9) - 1e-9]
        lines.append(f"  {algo}: mean {sum(gaps)/len(gaps):.3f}% "
                     f"(beats baseline on {len(beats)}/6: {', '.join(beats) or '-'})")
    lines.append(f"  alns policy: {policy.get('main_result_rule', '(none)')[:110]}...")
    lines.append("")

    # ---- Rush Hour hard ----
    hard_csv = results_dir / "rushhour_hard" / "gp_gep_hard_summary.csv"
    lines.append("== Rush Hour hard benchmark ==")
    if hard_csv.exists():
        for r in read_rows(hard_csv):
            lines.append(f"  {r['name']}: fitness {r['eval_fitness']} "
                         f"solved {r['solved']}/{r['puzzles']} expanded {r['expanded']}")
    else:
        lines.append(f"  MISSING: {hard_csv}")
    lines.append("")

    # ---- Ackley ----
    ackley = read_rows(results_dir / "summary" / "ackley_d10_summary.csv")
    lines.append(f"== Ackley (unchanged setup): {len(read_rows(results_dir / 'raw' / 'ackley_d10.csv'))} rows ==")
    for r in sorted(ackley, key=lambda r: float(r["best_value"])):
        lines.append(f"  {r['algorithm']}: best {r['best_value']}")

    text = "\n".join(lines)
    print(text)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + "\n")
    print(f"\nwritten to: {output}")
    sys.exit(0)


if __name__ == "__main__":
    main()
