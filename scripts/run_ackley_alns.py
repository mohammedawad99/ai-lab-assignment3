"""Run the ALNS-style variant on the Ackley function.

Usage:
    python scripts/run_ackley_alns.py [--iterations 1000] [--seed 42]
"""

import argparse
import sys
from pathlib import Path

# make "src" importable when the script is run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ackley.alns import run_ackley_alns
from src.ackley.csv_io import write_ackley_results


def main():
    parser = argparse.ArgumentParser(
        description="ALNS-style warm-up run on the Ackley function."
    )
    parser.add_argument("--dimension", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lower", type=float, default=-32.768)
    parser.add_argument("--upper", type=float, default=32.768)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--min-destroy", type=int, default=1)
    parser.add_argument("--max-destroy-fraction", type=float, default=0.4)
    parser.add_argument("--initial-temperature", type=float, default=10.0)
    parser.add_argument("--cooling-rate", type=float, default=0.995)
    parser.add_argument("--reaction-rate", type=float, default=0.2)
    parser.add_argument("--mutation-scale", type=float, default=1.0)
    parser.add_argument("--output", help="optional path for a one-row CSV")
    args = parser.parse_args()

    result = run_ackley_alns(
        dimension=args.dimension,
        iterations=args.iterations,
        seed=args.seed,
        lower=args.lower,
        upper=args.upper,
        timeout_sec=args.timeout,
        min_destroy=args.min_destroy,
        max_destroy_fraction=args.max_destroy_fraction,
        initial_temperature=args.initial_temperature,
        cooling_rate=args.cooling_rate,
        reaction_rate=args.reaction_rate,
        mutation_scale=args.mutation_scale,
    )

    print(f"algorithm: {result.algorithm}")
    print(f"dimension: {result.dimension}")
    print(f"seed: {result.seed}")
    print(f"iterations: {result.iterations}")
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
