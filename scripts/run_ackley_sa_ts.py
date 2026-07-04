"""Run Simulated Annealing and/or Tabu Search on the Ackley function.

Usage:
    python scripts/run_ackley_sa_ts.py --algorithm sa
    python scripts/run_ackley_sa_ts.py --algorithm tabu
    python scripts/run_ackley_sa_ts.py --algorithm both --output results/ackley_sa_ts.csv
"""

import argparse
import sys
from pathlib import Path

# make "src" importable when the script is run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ackley.csv_io import write_ackley_results
from src.ackley.simulated_annealing import run_ackley_simulated_annealing
from src.ackley.tabu_search import run_ackley_tabu_search


def main():
    parser = argparse.ArgumentParser(description="Run SA and/or Tabu Search on Ackley.")
    parser.add_argument("--algorithm", required=True, choices=["sa", "tabu", "both"])
    parser.add_argument("--dimension", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lower", type=float, default=-32.768)
    parser.add_argument("--upper", type=float, default=32.768)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--output", help="optional path for a CSV summary")
    args = parser.parse_args()

    common = dict(
        dimension=args.dimension, iterations=args.iterations,
        lower=args.lower, upper=args.upper, timeout_sec=args.timeout,
    )

    results = []
    if args.algorithm in ("sa", "both"):
        results.append(run_ackley_simulated_annealing(seed=args.seed, **common))
    if args.algorithm in ("tabu", "both"):
        # for "both" use seed+1 so the two runs do not share the same start
        tabu_seed = args.seed + 1 if args.algorithm == "both" else args.seed
        results.append(run_ackley_tabu_search(seed=tabu_seed, **common))

    for result in results:
        print(f"algorithm: {result.algorithm}")
        print(f"  dimension: {result.dimension}")
        print(f"  seed: {result.seed}")
        print(f"  iterations: {result.iterations}")
        print(f"  best_value: {result.best_value:.6f}")
        print(f"  distance_from_origin: {result.distance_from_origin:.6f}")
        print(f"  elapsed_time: {result.elapsed_time:.4f}")
        print(f"  cpu_time: {result.cpu_time:.4f}")

    if args.output:
        write_ackley_results(results, args.output)
        print(f"csv written to: {args.output}")

    sys.exit(0)


if __name__ == "__main__":
    main()
