"""Run the ILS evidence suite on the six official CVRP instances.

Uses exactly the canonical final-experiment settings from
configs/final_experiment_plan.json — same instances, same seeds, same
per-instance budgets and timeouts as the six-algorithm comparison — so the
ILS rows are directly comparable to the committed evidence. Outputs:

    results/final_experiments/ils/raw/cvrp_ils_runs.csv
    results/final_experiments/ils/summary/cvrp_ils_summary.csv
    results/final_experiments/ils/cvrp_ils_manifest.json
    results/final_experiments/ils/cvrp_solutions/<instance>_ils_seed<seed>.txt

and, with --refresh-evidence, copies the three small files to
report/evidence/ (the committed snapshots cited by the report).

Usage:
    python scripts/run_ils_evidence.py [--plan configs/final_experiment_plan.json]
        [--tuned-cvrp configs/tuned_cvrp_settings.json] [--refresh-evidence]
"""

import argparse
import csv
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# make "src" importable when the script is run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cvrp.io_cvrplib import parse_cvrplib
from src.experiments.cvrp_experiments import (
    CVRP_FIELDNAMES,
    run_one_cvrp_algorithm,
)
from src.experiments.official_benchmarks import load_bks_table
from src.experiments.summary import (
    CVRP_SUMMARY_FIELDNAMES,
    summarize_cvrp_rows,
    write_summary_csv,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(
        description="Run the canonical ILS evidence suite (six instances, "
                    "plan seeds/budgets/timeouts).")
    parser.add_argument("--plan", default="configs/final_experiment_plan.json",
                        help="final experiment plan JSON")
    parser.add_argument("--tuned-cvrp", default="configs/tuned_cvrp_settings.json",
                        help="tuned settings JSON (optional 'ils' section)")
    parser.add_argument("--output-dir", default="results/final_experiments/ils",
                        help="output directory for raw/summary/manifest files")
    parser.add_argument("--refresh-evidence", action="store_true",
                        help="copy the snapshots into report/evidence/")
    args = parser.parse_args()

    plan = json.loads(Path(args.plan).read_text())
    cvrp_plan = plan["cvrp"]
    data_dir = REPO_ROOT / cvrp_plan["data_dir"]
    seeds = cvrp_plan["seeds"]
    instances = cvrp_plan["instances"]
    per_instance = cvrp_plan["per_instance"]
    bks = load_bks_table(REPO_ROOT / cvrp_plan["bks_path"])

    tuned = {}
    tuned_path = Path(args.tuned_cvrp)
    if tuned_path.exists():
        tuned = json.loads(tuned_path.read_text())

    missing = [name for name in instances
               if not (data_dir / f"{name}.vrp").exists()]
    if missing:
        print("missing official .vrp files (user-provided, see README):",
              file=sys.stderr)
        for name in missing:
            print(f"  {data_dir / (name + '.vrp')}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw"
    summary_dir = output_dir / "summary"
    raw_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    suite_start = time.perf_counter()
    for name in instances:
        instance_path = data_dir / f"{name}.vrp"
        try:
            instance = parse_cvrplib(instance_path)
        except (OSError, ValueError) as e:
            sys.exit(f"input error: {instance_path}: {e}")
        budget = per_instance[name]["budget"]
        timeout_sec = per_instance[name]["timeout_sec"]
        for seed in seeds:
            row = run_one_cvrp_algorithm(
                instance, instance_path, "ils", seed, budget, timeout_sec,
                output_dir=output_dir, bks_cost=bks.get(name), tuned=tuned)
            rows.append(row)
            print(f"{name} seed {seed}: cost {row['best_cost']} "
                  f"gap {row['gap_percent']}% feasible {row['feasible']} "
                  f"elapsed {row['elapsed_time']}s (budget {budget}, "
                  f"timeout {timeout_sec}s)", flush=True)

    raw_path = raw_dir / "cvrp_ils_runs.csv"
    with open(raw_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CVRP_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = summarize_cvrp_rows(rows)
    summary_path = summary_dir / "cvrp_ils_summary.csv"
    write_summary_csv(summary_path, summary_rows, CVRP_SUMMARY_FIELDNAMES)

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": "python scripts/run_ils_evidence.py",
        "plan": str(args.plan),
        "algorithm": "cvrp_ils",
        "instances": instances,
        "seeds": seeds,
        "per_instance": per_instance,
        "ils_settings": tuned.get("ils", {
            "note": "library defaults: kick ~10% of customers, "
                    "worse_tolerance 0.02, restart_after 20, "
                    "candidate_list_k 10 on instances with >= 60 customers"}),
        "rows": len(rows),
        "feasible_rows": sum(1 for r in rows
                             if str(r["feasible"]).lower() == "true"),
        "total_elapsed_sec": round(time.perf_counter() - suite_start, 1),
    }
    manifest_path = output_dir / "cvrp_ils_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"raw rows: {raw_path} ({len(rows)} rows)")
    print(f"summary: {summary_path}")
    print(f"manifest: {manifest_path}")

    if args.refresh_evidence:
        evidence = REPO_ROOT / "report" / "evidence"
        evidence.mkdir(parents=True, exist_ok=True)
        for source, target in [
            (raw_path, evidence / "cvrp_ils_runs.csv"),
            (summary_path, evidence / "cvrp_ils_summary.csv"),
            (manifest_path, evidence / "cvrp_ils_manifest.json"),
        ]:
            shutil.copyfile(source, target)
            print(f"evidence snapshot: {target.relative_to(REPO_ROOT)}")

    infeasible = [r for r in rows if str(r["feasible"]).lower() != "true"]
    sys.exit(1 if infeasible else 0)


if __name__ == "__main__":
    main()
