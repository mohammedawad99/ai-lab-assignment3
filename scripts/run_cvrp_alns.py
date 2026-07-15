"""Run ALNS on one CVRP instance.

Usage:
    python scripts/run_cvrp_alns.py --instance examples/tiny_cvrp.vrp
"""

import argparse
import sys
from pathlib import Path

# make "src" importable when the script is run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cvrp.distance import build_distance_matrix
from src.cvrp.io_cvrplib import parse_cvrplib
from src.cvrp.solution_io import format_solution
from src.cvrp.solvers.alns import run_cvrp_alns


def main():
    parser = argparse.ArgumentParser(description="Run ALNS on a CVRP instance.")
    parser.add_argument("--instance", required=True, help="path to a CVRPLIB .vrp file")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--min-removal", type=int, default=1)
    parser.add_argument("--max-removal-fraction", type=float, default=0.3)
    parser.add_argument("--initial-temperature", type=float, default=100.0)
    parser.add_argument("--cooling-rate", type=float, default=0.995)
    parser.add_argument("--reaction-rate", type=float, default=0.2)
    parser.add_argument("--output", help="optional path to write the best solution")
    parser.add_argument("--include-unused-vehicles", action="store_true",
                        help="also print '0 0' lines for unused vehicles")
    args = parser.parse_args()

    try:
        instance = parse_cvrplib(args.instance)
    except (OSError, ValueError) as e:
        sys.exit(f"input error: {e}")
    result = run_cvrp_alns(
        instance,
        iterations=args.iterations,
        seed=args.seed,
        timeout_sec=args.timeout,
        min_removal=args.min_removal,
        max_removal_fraction=args.max_removal_fraction,
        initial_temperature=args.initial_temperature,
        cooling_rate=args.cooling_rate,
        reaction_rate=args.reaction_rate,
    )

    matrix = build_distance_matrix(instance)
    text = format_solution(result.best_solution, matrix,
                           include_unused_vehicles=args.include_unused_vehicles,
                           vehicle_count=instance.vehicle_count)

    print(f"algorithm: {result.algorithm}")
    print(f"instance: {result.instance_name}")
    print(f"seed: {result.seed}")
    print(f"iterations: {result.iterations}")
    print(f"initial_cost: {result.initial_cost:.2f}")
    print(f"best_cost: {result.best_cost:.2f}")
    print(f"feasible: {'yes' if result.feasible else 'no'}")
    print(f"elapsed_time: {result.elapsed_time:.4f}")
    print(f"cpu_time: {result.cpu_time:.4f}")
    last = result.convergence[-1]
    print("final operator weights:")
    print(f"  random_removal: {last['random_removal_weight']:.3f}")
    print(f"  worst_removal: {last['worst_removal_weight']:.3f}")
    print(f"  greedy_repair: {last['greedy_repair_weight']:.3f}")
    print(f"  regret2_repair: {last['regret2_repair_weight']:.3f}")
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
