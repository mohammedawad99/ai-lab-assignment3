"""Run the multi-stage CVRP baseline on one instance.

Usage:
    python scripts/run_cvrp_baseline.py --instance examples/tiny_cvrp.vrp \
        [--output results/tiny_baseline_solution.txt] [--include-unused-vehicles]
"""

import argparse
import sys
from pathlib import Path

# make "src" importable when the script is run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cvrp.baseline import build_multistage_baseline
from src.cvrp.distance import build_distance_matrix
from src.cvrp.io_cvrplib import parse_cvrplib
from src.cvrp.solution_io import format_solution


def main():
    parser = argparse.ArgumentParser(description="Run the multi-stage CVRP baseline.")
    parser.add_argument("--instance", required=True, help="path to a CVRPLIB .vrp file")
    parser.add_argument("--output", help="optional path to write the solution file")
    parser.add_argument("--include-unused-vehicles", action="store_true",
                        help="also print '0 0' lines for unused vehicles")
    args = parser.parse_args()

    try:
        instance = parse_cvrplib(args.instance)
    except (OSError, ValueError) as e:
        sys.exit(f"input error: {e}")
    result = build_multistage_baseline(instance)
    matrix = build_distance_matrix(instance)

    text = format_solution(result.solution, matrix,
                           include_unused_vehicles=args.include_unused_vehicles,
                           vehicle_count=instance.vehicle_count)

    print(f"instance: {instance.name}")
    print(f"construction_cost: {result.construction_cost:.2f}")
    print(f"after_2opt_cost: {result.after_2opt_cost:.2f}")
    print(f"final_cost: {result.final_cost:.2f}")
    print(f"feasible: {'yes' if result.feasible else 'no'}")
    for error in result.errors:
        print(f"  error: {error}")
    print("solution:")
    print(text)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n")
        print(f"solution written to: {output_path}")

    sys.exit(0 if result.feasible else 1)


if __name__ == "__main__":
    main()
