"""Run the GA Island Model on the Ackley function.

Usage:
    python scripts/run_ackley_ga_island.py [--generations 100] [--islands 4] [--seed 42]
"""

import argparse
import sys
from pathlib import Path

# make "src" importable when the script is run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ackley.csv_io import write_ackley_results
from src.ackley.ga_island import run_ackley_ga_island


def main():
    parser = argparse.ArgumentParser(description="Run GA-Island on the Ackley function.")
    parser.add_argument("--dimension", type=int, default=10)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--population-size", type=int, default=30)
    parser.add_argument("--islands", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lower", type=float, default=-32.768)
    parser.add_argument("--upper", type=float, default=32.768)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--crossover-rate", type=float, default=0.8)
    parser.add_argument("--mutation-rate", type=float, default=0.2)
    parser.add_argument("--mutation-scale", type=float, default=1.0)
    parser.add_argument("--migration-interval", type=int, default=20)
    parser.add_argument("--migrants", type=int, default=1)
    parser.add_argument("--output", help="optional path for a one-row CSV")
    args = parser.parse_args()

    result = run_ackley_ga_island(
        dimension=args.dimension,
        generations=args.generations,
        population_size=args.population_size,
        islands=args.islands,
        seed=args.seed,
        lower=args.lower,
        upper=args.upper,
        timeout_sec=args.timeout,
        crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate,
        mutation_scale=args.mutation_scale,
        migration_interval=args.migration_interval,
        migrants=args.migrants,
    )

    print(f"algorithm: {result.algorithm}")
    print(f"dimension: {result.dimension}")
    print(f"seed: {result.seed}")
    print(f"generations: {result.iterations}")
    print(f"best_value: {result.best_value:.6f}")
    print(f"distance_from_origin: {result.distance_from_origin:.6f}")
    print(f"elapsed_time: {result.elapsed_time:.4f}")
    print(f"cpu_time: {result.cpu_time:.4f}")
    print("best_x: " + " ".join(f"{v:.3f}" for v in result.best_x))

    if args.output:
        write_ackley_results([result], args.output)
        print(f"csv written to: {args.output}")

    sys.exit(0)


if __name__ == "__main__":
    main()
