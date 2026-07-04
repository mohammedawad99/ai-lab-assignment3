"""Run the final experiment suite from the plan, resumably.

Usage:
    python scripts/run_final_experiments.py
    python scripts/run_final_experiments.py --no-resume
    python scripts/run_final_experiments.py --skip-cvrp --skip-ackley
"""

import argparse
import sys
from pathlib import Path

# make "src" importable when the script is run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments.final_execution import run_final_experiment_suite


def main():
    parser = argparse.ArgumentParser(description="Run the final experiment suite.")
    parser.add_argument("--plan", default="configs/final_experiment_plan.json")
    parser.add_argument("--no-resume", action="store_true",
                        help="rerun everything even if raw CSVs already exist")
    parser.add_argument("--skip-cvrp", action="store_true")
    parser.add_argument("--skip-ackley", action="store_true")
    parser.add_argument("--skip-rushhour", action="store_true")
    args = parser.parse_args()

    try:
        manifest = run_final_experiment_suite(
            plan_path=args.plan,
            resume=not args.no_resume,
            run_cvrp=not args.skip_cvrp,
            run_ackley=not args.skip_ackley,
            run_rushhour=not args.skip_rushhour,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    except Exception as exc:  # unexpected runtime failure
        print(f"final experiment suite failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"plan: {manifest['plan_path']}")
    print(f"resume: {'yes' if manifest['resume'] else 'no'}")
    print(f"parts run: cvrp={manifest['run_cvrp']} "
          f"ackley={manifest['run_ackley']} rushhour={manifest['run_rushhour']}")

    for entry in manifest["cvrp_run_info"]:
        print(f"cvrp {entry['instance']}: rows={entry['rows']} "
              f"skipped_existing={entry['skipped_existing']} "
              f"budget={entry['budget']} timeout={entry['timeout_sec']} "
              f"elapsed={entry['elapsed_time']:.1f}s -> {entry['output_csv']}")
    if manifest["ackley_run_info"]:
        a = manifest["ackley_run_info"]
        print(f"ackley: rows={a['rows']} skipped_existing={a['skipped_existing']} "
              f"-> {a['output_csv']}")
    if manifest["rushhour_run_info"]:
        r = manifest["rushhour_run_info"]
        print(f"rushhour: rows={r['rows']} skipped_existing={r['skipped_existing']} "
              f"-> {r['output_csv']}")

    combined = manifest["aggregation_info"]["cvrp_combined"]
    print(f"combined cvrp csv: {combined['output_path']} "
          f"({combined['row_count']} rows, {combined['missing_count']} missing inputs)")
    for path in manifest["aggregation_info"]["summaries"]:
        print(f"summary: {path}")
    print(f"report assets: {len(manifest['asset_paths'])}")
    print(f"manifest: {manifest['manifest_path']}")
    sys.exit(0)


if __name__ == "__main__":
    main()
