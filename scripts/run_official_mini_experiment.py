"""Run a small first experiment on the official CVRP data, if placed.

Usage:
    python scripts/run_official_mini_experiment.py [--allow-missing-data]
"""

import argparse
import sys
from pathlib import Path

# make "src" importable when the script is run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments.official_benchmarks import DEFAULT_BKS_PATH, DEFAULT_OFFICIAL_CVRP_DIR
from src.experiments.official_mini_run import (
    OFFICIAL_MINI_DEFAULT_ALGORITHMS,
    OFFICIAL_MINI_DEFAULT_BUDGET,
    OFFICIAL_MINI_DEFAULT_SEEDS,
    OFFICIAL_MINI_DEFAULT_TIMEOUT_SEC,
    run_official_mini_experiment,
)
from src.experiments.summary import safe_float


def fail(message):
    print(message, file=sys.stderr)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="First official CVRP mini-run.")
    parser.add_argument("--data-dir", default=DEFAULT_OFFICIAL_CVRP_DIR)
    parser.add_argument("--bks", default=DEFAULT_BKS_PATH)
    parser.add_argument("--instances", nargs="+",
                        help="official instance names (default: smallest present)")
    parser.add_argument("--algorithms", nargs="+",
                        default=list(OFFICIAL_MINI_DEFAULT_ALGORITHMS))
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=list(OFFICIAL_MINI_DEFAULT_SEEDS))
    parser.add_argument("--budget", type=int, default=OFFICIAL_MINI_DEFAULT_BUDGET)
    parser.add_argument("--timeout", type=float, default=OFFICIAL_MINI_DEFAULT_TIMEOUT_SEC)
    parser.add_argument("--output-dir", default="results/official_mini")
    parser.add_argument("--include-unused-vehicles", action="store_true")
    parser.add_argument("--allow-missing-data", action="store_true",
                        help="skip cleanly when the official files are not placed yet")
    args = parser.parse_args()

    for name in args.algorithms:
        if name not in OFFICIAL_MINI_DEFAULT_ALGORITHMS:
            fail(f"unknown algorithm '{name}', "
                 f"known: {', '.join(OFFICIAL_MINI_DEFAULT_ALGORITHMS)}")

    try:
        result = run_official_mini_experiment(
            data_dir=args.data_dir,
            bks_path=args.bks,
            requested_instances=args.instances,
            algorithms=args.algorithms,
            seeds=args.seeds,
            budget=args.budget,
            timeout_sec=args.timeout,
            output_dir=args.output_dir,
            include_unused_vehicles=args.include_unused_vehicles,
            allow_missing_data=args.allow_missing_data,
        )
    except FileNotFoundError as exc:
        fail(str(exc))
    except ValueError as exc:
        fail(str(exc))

    print(f"ok: {result['ok']}")
    print(f"skipped: {result['skipped']}")
    print(f"reason: {result['reason'] or '(none)'}")
    print(f"selected instances: {', '.join(result['selected_instances']) or '(none)'}")
    print(f"missing instances: {', '.join(result['missing_instances']) or '(none)'}")
    print(f"algorithms: {' '.join(args.algorithms)}")
    print(f"seeds: {' '.join(str(s) for s in args.seeds)}")
    print(f"budget: {args.budget}")
    print(f"timeout: {args.timeout}")
    print(f"raw csv: {result['raw_csv'] or '(none)'}")
    print(f"summary csv: {result['summary_csv'] or '(none)'}")
    print(f"solution dir: {result['solution_dir'] or '(none)'}")
    print(f"raw rows: {len(result['raw_rows'])}")
    print(f"summary rows: {len(result['summary_rows'])}")

    if not result["skipped"]:
        for instance in result["selected_instances"]:
            costs = [c for c in (safe_float(row["best_cost"])
                                 for row in result["raw_rows"]
                                 if row["instance"] == instance)
                     if c is not None]
            if costs:
                print(f"best cost on {instance}: {min(costs):.4f}")

    sys.exit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
