"""Run the discretized LDS-style variant on the Ackley function.

Usage:
    python scripts/run_ackley_bnb_lds.py [--dimension 10] [--bins-per-dimension 7]
"""

import argparse
import sys
from pathlib import Path

# make "src" importable when the script is run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ackley.bnb_lds import run_ackley_bnb_lds
from src.ackley.csv_io import write_ackley_results


def main():
    parser = argparse.ArgumentParser(
        description="Discretized LDS-style warm-up run on the Ackley function."
    )
    parser.add_argument("--dimension", type=int, default=10)
    parser.add_argument("--bins-per-dimension", type=int, default=7)
    parser.add_argument("--max-discrepancy", type=int, default=3)
    parser.add_argument("--max-nodes", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lower", type=float, default=-32.768)
    parser.add_argument("--upper", type=float, default=32.768)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--output", help="optional path for a one-row CSV")
    args = parser.parse_args()

    result = run_ackley_bnb_lds(
        dimension=args.dimension,
        bins_per_dimension=args.bins_per_dimension,
        max_discrepancy=args.max_discrepancy,
        max_nodes=args.max_nodes,
        seed=args.seed,
        lower=args.lower,
        upper=args.upper,
        timeout_sec=args.timeout,
    )

    print(f"algorithm: {result.algorithm}")
    print(f"dimension: {result.dimension}")
    print(f"seed: {result.seed}")
    print(f"expanded_nodes: {result.iterations}")
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
