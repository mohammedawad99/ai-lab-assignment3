"""Unified experiment runner for CVRP and Ackley algorithms.

Usage examples:
    python scripts/run_experiments.py --part cvrp --instances examples/tiny_cvrp.vrp \
        --algorithms sa tabu --seeds 42 --budget 20 --timeout 5 --output results/cvrp_smoke.csv
    python scripts/run_experiments.py --part ackley --algorithms sa tabu --seeds 42 \
        --budget 20 --timeout 5 --dimension 4 --output results/ackley_smoke.csv
    python scripts/run_experiments.py --part both --instances examples/tiny_cvrp.vrp \
        --algorithms sa tabu --seeds 42 --budget 20 --timeout 5 --dimension 4 \
        --output-dir results/smoke_experiments
"""

import argparse
import sys
from pathlib import Path

# make "src" importable when the script is run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments.ackley_experiments import ACKLEY_FIELDNAMES, run_ackley_experiments
from src.experiments.csv_utils import write_dict_rows
from src.experiments.cvrp_experiments import CVRP_FIELDNAMES, run_cvrp_experiments

# the six required algorithms, shared by both parts
REQUIRED_SIX = ["sa", "tabu", "aco", "ga_island", "alns", "bnb_lds"]


def fail(message):
    print(message, file=sys.stderr)
    sys.exit(1)


def print_best_cvrp_rows(rows):
    by_instance = {}
    for row in rows:
        try:
            cost = float(row["best_cost"])
        except (ValueError, TypeError):
            continue
        name = row["instance"]
        if name not in by_instance or cost < float(by_instance[name]["best_cost"]):
            by_instance[name] = row
    for name, row in sorted(by_instance.items()):
        print(f"best cvrp on {name}: {row['algorithm']} seed {row['seed']} "
              f"cost {row['best_cost']} (feasible: {row['feasible']})")


def print_best_ackley_row(rows):
    best = None
    for row in rows:
        try:
            value = float(row["best_value"])
        except (ValueError, TypeError):
            continue
        if best is None or value < float(best["best_value"]):
            best = row
    if best is not None:
        print(f"best ackley: {best['algorithm']} seed {best['seed']} "
              f"value {best['best_value']}")


def main():
    parser = argparse.ArgumentParser(description="Run CVRP and/or Ackley experiments.")
    parser.add_argument("--part", required=True, choices=["cvrp", "ackley", "both"])
    parser.add_argument("--instances", nargs="+", help="CVRP instance paths")
    parser.add_argument("--algorithms", nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--dimension", type=int, default=10)
    parser.add_argument("--lower", type=float, default=-32.768)
    parser.add_argument("--upper", type=float, default=32.768)
    parser.add_argument("--include-baseline", action="store_true",
                        help="also run the multi-stage baseline reference (CVRP)")
    parser.add_argument("--include-random-search", action="store_true",
                        help="also run the random-search sanity baseline (Ackley)")
    parser.add_argument("--include-unused-vehicles", action="store_true")
    parser.add_argument("--output", help="CSV path for part=cvrp or part=ackley")
    parser.add_argument("--output-dir", help="directory for part=both")
    args = parser.parse_args()

    for name in args.algorithms:
        if name not in REQUIRED_SIX:
            fail(f"unknown algorithm '{name}', known: {', '.join(REQUIRED_SIX)}")
    if args.part in ("cvrp", "both") and not args.instances:
        fail("--instances is required for part=cvrp and part=both")
    if args.part in ("cvrp", "ackley") and not args.output:
        fail("--output is required for part=cvrp and part=ackley")
    if args.part == "both" and not args.output_dir:
        fail("--output-dir is required for part=both")

    cvrp_algorithms = (["baseline"] if args.include_baseline else []) + args.algorithms
    ackley_algorithms = (["random_search"] if args.include_random_search else []) + args.algorithms

    if args.part == "cvrp":
        output = Path(args.output)
        rows = run_cvrp_experiments(
            args.instances, cvrp_algorithms, args.seeds, budget=args.budget,
            timeout_sec=args.timeout, output_dir=output.parent,
            include_unused_vehicles=args.include_unused_vehicles,
        )
        write_dict_rows(output, rows, CVRP_FIELDNAMES)
        print(f"cvrp rows written: {len(rows)} -> {output}")
        print_best_cvrp_rows(rows)

    elif args.part == "ackley":
        output = Path(args.output)
        rows = run_ackley_experiments(
            ackley_algorithms, args.seeds, budget=args.budget,
            timeout_sec=args.timeout, dimension=args.dimension,
            lower=args.lower, upper=args.upper,
        )
        write_dict_rows(output, rows, ACKLEY_FIELDNAMES)
        print(f"ackley rows written: {len(rows)} -> {output}")
        print_best_ackley_row(rows)

    else:  # both
        output_dir = Path(args.output_dir)
        cvrp_rows = run_cvrp_experiments(
            args.instances, cvrp_algorithms, args.seeds, budget=args.budget,
            timeout_sec=args.timeout, output_dir=output_dir,
            include_unused_vehicles=args.include_unused_vehicles,
        )
        cvrp_csv = output_dir / "cvrp_results.csv"
        write_dict_rows(cvrp_csv, cvrp_rows, CVRP_FIELDNAMES)
        print(f"cvrp rows written: {len(cvrp_rows)} -> {cvrp_csv}")
        print_best_cvrp_rows(cvrp_rows)

        ackley_rows = run_ackley_experiments(
            ackley_algorithms, args.seeds, budget=args.budget,
            timeout_sec=args.timeout, dimension=args.dimension,
            lower=args.lower, upper=args.upper,
        )
        ackley_csv = output_dir / "ackley_results.csv"
        write_dict_rows(ackley_csv, ackley_rows, ACKLEY_FIELDNAMES)
        print(f"ackley rows written: {len(ackley_rows)} -> {ackley_csv}")
        print_best_ackley_row(ackley_rows)

    sys.exit(0)


if __name__ == "__main__":
    main()
