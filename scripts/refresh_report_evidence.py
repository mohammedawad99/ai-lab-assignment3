"""Refresh the small committed evidence snapshots under report/evidence/
from the local final results under results/final_experiments/.

Derives the two policy-dependent CVRP summaries (the ALNS selection rule from
configs/tuned_cvrp_settings.json is applied here, matching the figures):
  - cvrp_all_policy_effective_summary.csv: per-instance summary rows with the
    effective ALNS variant only, renamed to cvrp_alns
  - cvrp_algorithm_mean_gaps.csv: per-algorithm best-of-seeds gap per instance
    plus the mean over the six instances

and copies the rest verbatim (CVRP summary, Ackley summary, hard Rush Hour
files, execution manifest, final comparison summary).

Usage:
    .venv/bin/python scripts/refresh_report_evidence.py
"""

import csv
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results" / "final_experiments"
EVIDENCE = REPO_ROOT / "report" / "evidence"
INSTANCES = ["P-n16-k8", "E-n22-k4", "A-n32-k5", "A-n80-k10",
             "X-n101-k25", "M-n200-k17"]

COPIES = [
    ("summary/cvrp_all_summary.csv", "cvrp_all_summary.csv"),
    ("summary/ackley_d10_summary.csv", "ackley_d10_summary.csv"),
    ("raw/gp_gep_comparison_runs.csv", "gp_gep_comparison_runs.csv"),
    ("rushhour_hard/manual_heuristics_summary.csv", "rushhour_hard_manual_summary.csv"),
    ("rushhour_hard/gp_gep_hard_summary.csv", "rushhour_hard_gp_gep_summary.csv"),
    ("rushhour_hard/hard_benchmark_manifest.json", "rushhour_hard_manifest.json"),
    ("final_execution_manifest.json", "final_execution_manifest.json"),
    ("final_v3_summary.txt", "final_v3_summary.txt"),
]


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    with open(REPO_ROOT / "configs" / "tuned_cvrp_settings.json") as f:
        excluded = set(json.load(f).get("alns_policy", {})
                       .get("enhanced_excluded_instances", []))

    missing = [src for src, _ in COPIES if not (RESULTS / src).exists()]
    if missing:
        print("missing result files, refusing to refresh evidence:")
        for src in missing:
            print(f"  {RESULTS / src}")
        sys.exit(1)

    for src, dst in COPIES:
        shutil.copyfile(RESULTS / src, EVIDENCE / dst)
        print(f"copied {src} -> report/evidence/{dst}")

    # hard Rush Hour config snapshot comes from configs/, not from results
    shutil.copyfile(REPO_ROOT / "configs" / "rushhour_hard_benchmark.json",
                    EVIDENCE / "rushhour_hard_config.json")
    print("copied configs/rushhour_hard_benchmark.json -> "
          "report/evidence/rushhour_hard_config.json")

    # ---- policy-effective per-instance summary ----
    summary = read_rows(RESULTS / "summary" / "cvrp_all_summary.csv")
    effective = []
    for row in summary:
        algo, inst = row["algorithm"], row["instance"]
        if algo == "cvrp_alns" and inst not in excluded:
            continue
        if algo == "cvrp_alns_enhanced":
            if inst in excluded:
                continue
            row = dict(row, algorithm="cvrp_alns")
        effective.append(row)
    eff_path = EVIDENCE / "cvrp_all_policy_effective_summary.csv"
    write_rows(eff_path, effective)
    print(f"derived {eff_path.relative_to(REPO_ROOT)} ({len(effective)} rows)")

    # ---- per-algorithm mean of best-of-seeds gaps ----
    best = {}
    for row in effective:
        best.setdefault(row["algorithm"], {})[row["instance"]] = \
            float(row["best_gap_percent"])
    gap_rows = []
    for algo, gaps in best.items():
        row = {"algorithm": algo,
               "mean_best_gap_percent":
                   f"{sum(gaps[i] for i in INSTANCES) / len(INSTANCES):.3f}"}
        row.update({i: f"{gaps[i]:.4f}" for i in INSTANCES})
        gap_rows.append(row)
    gap_rows.sort(key=lambda r: float(r["mean_best_gap_percent"]))
    gaps_path = EVIDENCE / "cvrp_algorithm_mean_gaps.csv"
    write_rows(gaps_path, gap_rows)
    print(f"derived {gaps_path.relative_to(REPO_ROOT)} ({len(gap_rows)} algorithms)")

    # the superseded v2 comparison is replaced by final_v3_summary.txt
    old = EVIDENCE / "final_v2_summary.txt"
    if old.exists():
        old.unlink()
        print("removed superseded report/evidence/final_v2_summary.txt")


if __name__ == "__main__":
    main()
