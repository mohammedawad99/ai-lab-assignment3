"""Evaluate baseline Rush Hour heuristics on a puzzle set, with safety caps.

Usage:
    python scripts/evaluate_rushhour_heuristics.py --puzzles examples/rushhour_eval_puzzles.txt \
        [--heuristics zero blocking blocking_distance] [--max-nodes-per-puzzle N] \
        [--max-time-per-puzzle SEC] [--max-total-time SEC] [--output results/rushhour_heuristics.csv]
"""

import argparse
import sys
from pathlib import Path

# make "src" importable when the script is run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rushhour.evaluator import (
    evaluate_heuristic,
    fitness_from_evaluation,
    write_evaluation_csv,
)
from src.rushhour.heuristics import BASELINE_HEURISTICS
from src.rushhour.puzzle_sets import load_puzzle_set


def main():
    parser = argparse.ArgumentParser(description="Evaluate Rush Hour heuristics safely.")
    parser.add_argument("--puzzles", required=True, help="path to a puzzle set file")
    parser.add_argument("--heuristics", nargs="+",
                        default=["zero", "blocking", "blocking_distance"],
                        help="baseline heuristic names to evaluate")
    parser.add_argument("--max-nodes-per-puzzle", type=int, default=10000)
    parser.add_argument("--max-time-per-puzzle", type=float, default=1.0)
    parser.add_argument("--max-total-time", type=float, default=10.0)
    parser.add_argument("--output", help="optional path for a CSV summary")
    parser.add_argument("--per-puzzle-output",
                        help="optional CSV with one row per puzzle per heuristic")
    args = parser.parse_args()

    for name in args.heuristics:
        if name not in BASELINE_HEURISTICS:
            known = ", ".join(sorted(BASELINE_HEURISTICS))
            print(f"unknown heuristic '{name}', known: {known}", file=sys.stderr)
            sys.exit(1)

    puzzles = load_puzzle_set(args.puzzles)
    print(f"loaded {len(puzzles)} puzzles from {args.puzzles}")

    evaluations = []
    for name in args.heuristics:
        evaluation = evaluate_heuristic(
            name, BASELINE_HEURISTICS[name], puzzles,
            max_nodes_per_puzzle=args.max_nodes_per_puzzle,
            max_time_per_puzzle_sec=args.max_time_per_puzzle,
            max_total_time_sec=args.max_total_time,
        )
        evaluations.append(evaluation)
        fitness = fitness_from_evaluation(evaluation)
        print(f"{name}: solved {evaluation.solved_count}/{evaluation.puzzle_count} "
              f"(rate {evaluation.solved_rate:.2f}), "
              f"expanded {evaluation.total_expanded_nodes}, "
              f"cost {evaluation.total_cost}, fitness {fitness:.0f}")

    if args.output:
        write_evaluation_csv(evaluations, args.output)
        print(f"csv written to: {args.output}")

    if args.per_puzzle_output:
        import csv
        path = Path(args.per_puzzle_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["puzzle_id", "heuristic_name", "solved",
                             "solution_length", "expanded_nodes",
                             "generated_nodes", "runtime_seconds", "error"])
            for evaluation in evaluations:
                for r in evaluation.puzzle_results:
                    writer.writerow([
                        r.puzzle_index, evaluation.heuristic_name, r.solved,
                        r.cost if r.solved else "", r.expanded_nodes,
                        r.generated_nodes, f"{r.elapsed_time:.6f}",
                        "" if r.solved else (r.stopped_reason or ""),
                    ])
        print(f"per-puzzle csv written to: {path}")

    sys.exit(0)


if __name__ == "__main__":
    main()
