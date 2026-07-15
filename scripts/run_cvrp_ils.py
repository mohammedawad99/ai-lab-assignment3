"""Run Iterated Local Search (ILS) on one CVRP instance.

Usage:
    python scripts/run_cvrp_ils.py --instance examples/tiny_cvrp.vrp
    python scripts/run_cvrp_ils.py --instance data/official_cvrp/A-n32-k5.vrp \
        --iterations 500 --seed 42 --timeout 60
"""

import argparse
import sys
from pathlib import Path

# make "src" importable when the script is run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cvrp.distance import build_distance_matrix
from src.cvrp.io_cvrplib import parse_cvrplib
from src.cvrp.solution_io import format_solution
from src.cvrp.solvers.ils import run_cvrp_ils


def main():
    parser = argparse.ArgumentParser(
        description="Run Iterated Local Search on a CVRP instance.")
    parser.add_argument("--instance", required=True, help="path to a CVRPLIB .vrp file")
    parser.add_argument("--iterations", type=int, default=100,
                        help="ILS iterations (perturb + local search cycles)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=10.0,
                        help="wall-clock time limit in seconds")
    parser.add_argument("--kick-strength", type=int, default=None,
                        help="customers removed per perturbation "
                             "(default: ~10%% of the customers)")
    parser.add_argument("--worse-tolerance", type=float, default=0.02,
                        help="relative threshold for accepting worse local optima")
    parser.add_argument("--restart-after", type=int, default=20,
                        help="non-improving iterations before restarting from best")
    parser.add_argument("--candidate-list-k", type=int, default=10,
                        help="k-nearest candidate list size for the local search "
                             "on large instances (0 disables pruning)")
    parser.add_argument("--output", help="optional path to write the best solution")
    parser.add_argument("--include-unused-vehicles", action="store_true",
                        help="also print '0 0' lines for unused vehicles")
    args = parser.parse_args()

    instance = parse_cvrplib(args.instance)
    result = run_cvrp_ils(
        instance, iterations=args.iterations, seed=args.seed,
        timeout_sec=args.timeout, kick_strength=args.kick_strength,
        worse_tolerance=args.worse_tolerance, restart_after=args.restart_after,
        candidate_list_k=args.candidate_list_k or None,
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
    print(f"accepted_worse_count: {result.accepted_worse_count}")
    print(f"final_kick_strength: {result.perturbation_strength}")
    print(f"restarts: {result.restarts}")
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
