"""Run the GA Island Model on one CVRP instance.

Usage:
    python scripts/run_cvrp_ga_island.py --instance examples/tiny_cvrp.vrp
"""

import argparse
import sys
from pathlib import Path

# make "src" importable when the script is run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cvrp.distance import build_distance_matrix
from src.cvrp.io_cvrplib import parse_cvrplib
from src.cvrp.solution_io import format_solution
from src.cvrp.solvers.ga_island import run_cvrp_ga_island


def main():
    parser = argparse.ArgumentParser(description="Run GA-Island on a CVRP instance.")
    parser.add_argument("--instance", required=True, help="path to a CVRPLIB .vrp file")
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--population-size", type=int, default=30)
    parser.add_argument("--islands", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--crossover-rate", type=float, default=0.8)
    parser.add_argument("--mutation-rate", type=float, default=0.15)
    parser.add_argument("--migration-interval", type=int, default=20)
    parser.add_argument("--migrants", type=int, default=1)
    parser.add_argument("--output", help="optional path to write the best solution")
    parser.add_argument("--include-unused-vehicles", action="store_true",
                        help="also print '0 0' lines for unused vehicles")
    args = parser.parse_args()

    instance = parse_cvrplib(args.instance)
    result = run_cvrp_ga_island(
        instance,
        generations=args.generations,
        population_size=args.population_size,
        islands=args.islands,
        seed=args.seed,
        timeout_sec=args.timeout,
        crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate,
        migration_interval=args.migration_interval,
        migrants=args.migrants,
    )

    matrix = build_distance_matrix(instance)
    text = format_solution(result.best_solution, matrix,
                           include_unused_vehicles=args.include_unused_vehicles,
                           vehicle_count=instance.vehicle_count)

    print(f"algorithm: {result.algorithm}")
    print(f"instance: {result.instance_name}")
    print(f"seed: {result.seed}")
    print(f"generations: {result.iterations}")
    print(f"initial_cost: {result.initial_cost:.2f}")
    print(f"best_cost: {result.best_cost:.2f}")
    print(f"feasible: {'yes' if result.feasible else 'no'}")
    print(f"elapsed_time: {result.elapsed_time:.4f}")
    print(f"cpu_time: {result.cpu_time:.4f}")
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
