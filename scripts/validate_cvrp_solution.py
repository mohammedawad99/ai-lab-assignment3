"""Validate a CVRP solution file against a CVRPLIB instance file.

Usage:
    python scripts/validate_cvrp_solution.py --instance examples/tiny_cvrp.vrp \
        --solution examples/tiny_solution.txt [--include-unused-vehicles]
"""

import argparse
import sys
from pathlib import Path

# make "src" importable when the script is run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cvrp.cost import solution_cost
from src.cvrp.distance import build_distance_matrix
from src.cvrp.io_cvrplib import parse_cvrplib
from src.cvrp.solution_io import format_solution, parse_solution_file
from src.cvrp.validate import validate_solution


def main():
    parser = argparse.ArgumentParser(description="Validate a CVRP solution against an instance.")
    parser.add_argument("--instance", required=True, help="path to a CVRPLIB .vrp file")
    parser.add_argument("--solution", required=True, help="path to a solution file")
    parser.add_argument("--include-unused-vehicles", action="store_true",
                        help="also print '0 0' lines for unused vehicles")
    args = parser.parse_args()

    instance = parse_cvrplib(args.instance)
    solution = parse_solution_file(args.solution)
    matrix = build_distance_matrix(instance)
    cost = solution_cost(solution, matrix)
    result = validate_solution(instance, solution)

    print(f"instance: {instance.name}")
    print(f"cost: {cost:.2f}")
    print(f"feasible: {'yes' if result.feasible else 'no'}")
    for error in result.errors:
        print(f"  error: {error}")
    print("solution:")
    print(format_solution(solution, matrix,
                          include_unused_vehicles=args.include_unused_vehicles,
                          vehicle_count=instance.vehicle_count))

    sys.exit(0 if result.feasible else 1)


if __name__ == "__main__":
    main()
