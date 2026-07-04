"""Run the random-search sanity baseline on the Ackley function.

Usage:
    python scripts/run_ackley_random_search.py [--dimension 10] [--iterations 1000]
        [--seed 42] [--lower -32.768] [--upper 32.768]
        [--output results/ackley_random_search.csv]
"""

import argparse
import sys
from pathlib import Path

# make "src" importable when the script is run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ackley.csv_io import write_ackley_results
from src.ackley.random_search import run_random_search


def main():
    parser = argparse.ArgumentParser(
        description="Random-search sanity baseline for the Ackley warm-up."
    )
    parser.add_argument("--dimension", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lower", type=float, default=-32.768)
    parser.add_argument("--upper", type=float, default=32.768)
    parser.add_argument("--output", help="optional path for a one-row CSV")
    args = parser.parse_args()

    result = run_random_search(
        dimension=args.dimension,
        iterations=args.iterations,
        seed=args.seed,
        lower=args.lower,
        upper=args.upper,
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
