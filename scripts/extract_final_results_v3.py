"""Stage 11-C: compare the advanced-moves final rerun against the committed
Stage 10-H evidence.

Reads report/evidence/ (old, committed) and results/final_experiments/ (new
rerun with the Stage 11-B advanced local-search settings), applies the
pre-declared ALNS selection policy, and reports per-instance and per-algorithm
changes, B&B/LDS consistency, feasibility, Rush Hour, Ackley and runtime.

Optionally reads --old-results (a backup of the previous final_experiments
directory) for the old per-run runtimes, which the evidence CSVs do not carry.

Usage:
    python scripts/extract_final_results_v3.py
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
INSTANCES = ["P-n16-k8", "E-n22-k4", "A-n32-k5", "A-n80-k10", "X-n101-k25", "M-n200-k17"]
LARGE = ["A-n80-k10", "X-n101-k25", "M-n200-k17"]  # advanced gate active here


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def effective_rows(rows, excluded):
    """Apply the pre-declared ALNS rule: the assignment 'alns' result is the
    enhanced variant except on excluded instances."""
    out = []
    for row in rows:
        algo, inst = row["algorithm"], row["instance"]
        if algo == "cvrp_alns":
            keep = inst in excluded
        elif algo == "cvrp_alns_enhanced":
            keep = inst not in excluded
            algo = "cvrp_alns"
        else:
            keep = True
        if keep:
            out.append(dict(row, algorithm=algo))
    return out


def best_of_seeds(rows):
    """{algorithm: {instance: best feasible gap}}"""
    table = defaultdict(dict)
    for r in rows:
        if r["feasible"] != "True" or not r["gap_percent"]:
            continue
        gap = float(r["gap_percent"])
        cur = table[r["algorithm"]].get(r["instance"])
        if cur is None or gap < cur:
            table[r["algorithm"]][r["instance"]] = gap
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", default="results/final_experiments")
    parser.add_argument("--old-results",
                        default="results/final_experiments_pre11c_backup",
                        help="previous final_experiments dir for old runtimes")
    parser.add_argument("--output",
                        default="results/final_experiments/final_v3_summary.txt")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    lines = ["Stage 11-C final rerun summary (advanced CVRP moves)", ""]

    with open(REPO_ROOT / "configs" / "tuned_cvrp_settings.json") as f:
        tuned = json.load(f)
    excluded = set(tuned.get("alns_policy", {}).get("enhanced_excluded_instances", []))

    new_raw = read_rows(results_dir / "raw" / "cvrp_all_instances.csv")
    new_eff = effective_rows(new_raw, excluded)
    new_best = best_of_seeds(new_eff)

    # old committed evidence (per-algorithm per-instance best gaps)
    old_gaps = read_rows(REPO_ROOT / "report" / "evidence" / "cvrp_algorithm_mean_gaps.csv")
    old_by_algo = {r["algorithm"]: r for r in old_gaps}

    def old_best(inst):
        return min(float(r[inst]) for r in old_gaps)

    # ---- per-instance best gaps ----
    lines.append("== CVRP: project-best gap per instance (old evidence -> new rerun) ==")
    improved, regressed = [], []
    for inst in INSTANCES:
        new = min(t[inst] for t in new_best.values() if inst in t)
        prev = old_best(inst)
        delta = new - prev
        mark = "IMPROVED" if delta < -0.01 else ("same" if abs(delta) <= 0.01 else "REGRESSED")
        (improved if mark == "IMPROVED" else regressed if mark == "REGRESSED" else []).append(inst)
        best_algo = min((a for a in new_best if inst in new_best[a]),
                        key=lambda a: new_best[a][inst])
        lines.append(f"  {inst}: {prev:.4f}% -> {new:.4f}% ({delta:+.4f}) {mark} "
                     f"[{best_algo}]")
    lines.append(f"  improved: {', '.join(improved) or 'none'}; "
                 f"regressed: {', '.join(regressed) or 'none'}")
    lines.append("")

    # ---- per-algorithm means and beats-baseline ----
    lines.append("== CVRP: algorithm mean of best-of-seeds gaps (old -> new) ==")
    baseline_new = new_best.get("baseline", {})
    for algo in sorted(new_best):
        gaps = [new_best[algo][i] for i in INSTANCES if i in new_best[algo]]
        mean_new = sum(gaps) / len(gaps)
        old_row = old_by_algo.get(algo)
        old_mean = (f"{float(old_row['mean_best_gap_percent']):.3f}"
                    if old_row else "  n/a")
        beats = [i for i in INSTANCES
                 if new_best[algo].get(i, 1e9) < baseline_new.get(i, 1e9) - 1e-9]
        lines.append(f"  {algo}: {old_mean}% -> {mean_new:.3f}% "
                     f"(beats baseline on {len(beats)}/6)")
    lines.append("")

    # ---- ALNS / GA advanced impact per instance ----
    for algo, label in (("cvrp_alns", "ALNS (effective policy)"),
                        ("cvrp_ga_island", "GA-Island")):
        lines.append(f"== {label}: old -> new best gap per instance ==")
        old_row = old_by_algo.get(algo)
        for inst in INSTANCES:
            prev = float(old_row[inst])
            new = new_best.get(algo, {}).get(inst)
            gate = " (advanced gate ON)" if inst in LARGE else ""
            lines.append(f"  {inst}: {prev:.4f}% -> {new:.4f}% "
                         f"({new - prev:+.4f}){gate}")
        lines.append("")

    # ---- B&B/LDS consistency ----
    lines.append("== B&B/LDS: old evidence -> new rerun ==")
    old_bnb = old_by_algo["cvrp_bnb_lds"]
    consistent = True
    for inst in INSTANCES:
        prev = float(old_bnb[inst])
        new = new_best.get("cvrp_bnb_lds", {}).get(inst)
        if abs(new - prev) > 0.5:
            consistent = False
        lines.append(f"  {inst}: {prev:.4f}% -> {new:.4f}% ({new - prev:+.4f})")
    lines.append(f"  small-instance mode still effective: "
                 f"P {new_best['cvrp_bnb_lds']['P-n16-k8']:.4f}% and "
                 f"E {new_best['cvrp_bnb_lds']['E-n22-k4']:.4f}% "
                 f"({'PASS' if new_best['cvrp_bnb_lds']['P-n16-k8'] < 0.5 and new_best['cvrp_bnb_lds']['E-n22-k4'] < 0.5 else 'FAIL'})")
    lines.append(f"  values stable vs old evidence (+-0.5): "
                 f"{'PASS' if consistent else 'CHANGED (see deltas above)'}")
    lines.append("")

    # ---- feasibility ----
    feasible = sum(1 for r in new_raw if r["feasible"] == "True")
    errors = sum(1 for r in new_raw if r.get("errors"))
    lines.append(f"== CVRP feasibility: {feasible}/{len(new_raw)} rows feasible, "
                 f"{errors} rows with errors ==")
    lines.append("")

    # ---- runtime impact (old backup -> new) on the gated instances ----
    old_dir = Path(args.old_results)
    if (old_dir / "raw" / "cvrp_all_instances.csv").exists():
        old_raw = read_rows(old_dir / "raw" / "cvrp_all_instances.csv")
        lines.append("== runtime (mean elapsed sec per run, old -> new) on gated instances ==")
        for algo in ("cvrp_alns", "cvrp_alns_enhanced", "cvrp_ga_island"):
            for inst in LARGE:
                old_times = [float(r["elapsed_time"]) for r in old_raw
                             if r["algorithm"] == algo and r["instance"] == inst]
                new_times = [float(r["elapsed_time"]) for r in new_raw
                             if r["algorithm"] == algo and r["instance"] == inst]
                if old_times and new_times:
                    lines.append(f"  {algo:22s} {inst:12s} "
                                 f"{sum(old_times)/len(old_times):8.1f}s -> "
                                 f"{sum(new_times)/len(new_times):8.1f}s")
        lines.append("")

    # ---- Rush Hour hard ----
    lines.append("== Rush Hour hard benchmark ==")
    hard_csv = results_dir / "rushhour_hard" / "gp_gep_hard_summary.csv"
    if hard_csv.exists():
        for r in read_rows(hard_csv):
            lines.append(f"  {r['name']}: fitness {r['eval_fitness']} "
                         f"solved {r['solved']}/{r['puzzles']} expanded {r['expanded']}")
    else:
        lines.append(f"  MISSING: {hard_csv}")
    lines.append("")

    # ---- Ackley ----
    ackley_raw = read_rows(results_dir / "raw" / "ackley_d10.csv")
    ackley = read_rows(results_dir / "summary" / "ackley_d10_summary.csv")
    lines.append(f"== Ackley (unchanged setup): {len(ackley_raw)} rows ==")
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
