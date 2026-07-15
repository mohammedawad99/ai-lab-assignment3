"""Run the time-limited B&B/LDS solver on one CVRP instance.

Usage:
    python scripts/run_cvrp_bnb_lds.py --instance examples/tiny_cvrp.vrp
"""

import argparse
import sys
from pathlib import Path

# make "src" importable when the script is run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cvrp.distance import build_distance_matrix
from src.cvrp.io_cvrplib import parse_cvrplib
from src.cvrp.solution_io import format_solution
from src.cvrp.solvers.bnb_lds import run_cvrp_bnb_lds


def main():
    parser = argparse.ArgumentParser(description="Run B&B/LDS on a CVRP instance.")
    parser.add_argument("--instance", required=True, help="path to a CVRPLIB .vrp file")
    parser.add_argument("--max-discrepancy", type=int, default=3)
    parser.add_argument("--max-nodes", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--output", help="optional path to write the best solution")
    parser.add_argument("--include-unused-vehicles", action="store_true",
                        help="also print '0 0' lines for unused vehicles")
    args = parser.parse_args()

    try:
        instance = parse_cvrplib(args.instance)
    except (OSError, ValueError) as e:
        sys.exit(f"input error: {e}")
    result = run_cvrp_bnb_lds(
        instance,
        max_discrepancy=args.max_discrepancy,
        max_nodes=args.max_nodes,
        seed=args.seed,
        timeout_sec=args.timeout,
    )

    matrix = build_distance_matrix(instance)
    text = format_solution(result.best_solution, matrix,
                           include_unused_vehicles=args.include_unused_vehicles,
                           vehicle_count=instance.vehicle_count)

    print(f"algorithm: {result.algorithm}")
    print(f"instance: {result.instance_name}")
    print(f"seed: {result.seed}")
    print(f"expanded_nodes: {result.iterations}")
    print(f"initial_cost: {result.initial_cost:.2f}")
    print(f"best_cost: {result.best_cost:.2f}")
    print(f"feasible: {'yes' if result.feasible else 'no'}")
    print(f"elapsed_time: {result.elapsed_time:.4f}")
    print(f"cpu_time: {result.cpu_time:.4f}")
    last = result.convergence[-1]
    print("final search stats:")
    print(f"  pruned_by_bound: {last['pruned_by_bound']}")
    print(f"  pruned_by_discrepancy: {last['pruned_by_discrepancy']}")
    print(f"  completed_solutions: {last['completed_solutions']}")
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
