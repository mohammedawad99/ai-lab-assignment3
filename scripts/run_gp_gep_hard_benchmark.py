"""GP vs GEP on the hard Rush Hour benchmark, with manual-heuristic baselines.

Training uses the original 4 GP training puzzles plus the easiest few hard
puzzles; evaluation uses the full hard set. GP and GEP get identical
budgets, seeds and safety caps, and the same caps are used to score the
manual heuristics so the comparison row is fair.

Usage:
    python scripts/run_gp_gep_hard_benchmark.py --puzzles examples/rushhour_hard_eval.txt --seeds 42 43 44
"""

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.rushhour.evaluator import evaluate_heuristic, fitness_from_evaluation
from src.rushhour.gp_gep_comparison import (
    compare_gp_gep,
    write_comparison_summary_txt,
    write_run_summaries_csv,
)
from src.rushhour.heuristics import BASELINE_HEURISTICS
from src.rushhour.puzzle_sets import load_puzzle_set


def main():
    parser = argparse.ArgumentParser(description="GP vs GEP hard benchmark.")
    parser.add_argument("--puzzles", default="examples/rushhour_hard_eval.txt",
                        help="hard evaluation puzzle set")
    parser.add_argument("--train-puzzles", default="examples/rushhour_gp_train.txt")
    parser.add_argument("--train-hard-count", type=int, default=3,
                        help="how many of the easiest hard puzzles join training")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--population-size", type=int, default=30)
    parser.add_argument("--gp-max-depth", type=int, default=4)
    parser.add_argument("--gep-head-length", type=int, default=6)
    parser.add_argument("--max-nodes-per-puzzle", type=int, default=15000)
    parser.add_argument("--max-time-per-puzzle", type=float, default=2.0)
    parser.add_argument("--max-total-time", type=float, default=20.0)
    parser.add_argument("--output-dir", default="results/rushhour_hard")
    args = parser.parse_args()

    eval_puzzles = load_puzzle_set(args.puzzles)
    train_puzzles = (load_puzzle_set(args.train_puzzles)
                     + eval_puzzles[:args.train_hard_count])
    print(f"training puzzles: {len(train_puzzles)} "
          f"({args.train_puzzles} + first {args.train_hard_count} hard)")
    print(f"evaluation puzzles: {len(eval_puzzles)} ({args.puzzles})")

    caps = dict(max_nodes_per_puzzle=args.max_nodes_per_puzzle,
                max_time_per_puzzle_sec=args.max_time_per_puzzle,
                max_total_time_sec=args.max_total_time)

    summary = compare_gp_gep(
        train_puzzles, eval_puzzles, args.seeds,
        generations=args.generations, population_size=args.population_size,
        gp_max_depth=args.gp_max_depth, gep_head_length=args.gep_head_length,
        **caps,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_run_summaries_csv(summary, output_dir / "gp_gep_hard_runs.csv")
    write_comparison_summary_txt(summary, output_dir / "gp_gep_hard_summary.txt")

    # manual baselines on the same evaluation set with the same caps
    manual_rows = []
    for name in sorted(BASELINE_HEURISTICS):
        evaluation = evaluate_heuristic(name, BASELINE_HEURISTICS[name],
                                        eval_puzzles, **caps)
        manual_rows.append({
            "name": f"manual_{name}",
            "eval_fitness": fitness_from_evaluation(evaluation),
            "solved": evaluation.solved_count,
            "puzzles": evaluation.puzzle_count,
            "expanded": evaluation.total_expanded_nodes,
        })

    summary_csv = output_dir / "gp_gep_hard_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "eval_fitness", "solved", "puzzles",
                         "expanded", "best_expression"])
        for run in summary.gp_runs + summary.gep_runs:
            writer.writerow([f"{run.algorithm}_seed{run.seed}",
                             f"{run.eval_fitness:.0f}", run.eval_solved_count,
                             run.eval_puzzle_count,
                             run.eval_total_expanded_nodes,
                             run.best_expression])
        for row in manual_rows:
            writer.writerow([row["name"], f"{row['eval_fitness']:.0f}",
                             row["solved"], row["puzzles"], row["expanded"], ""])

    print("\n== hard benchmark summary (higher fitness is better) ==")
    print(f"{'name':>28} {'fitness':>9} {'solved':>7} {'expanded':>9}")
    for run in summary.gp_runs + summary.gep_runs:
        print(f"{run.algorithm + '_seed' + str(run.seed):>28} "
              f"{run.eval_fitness:>9.0f} "
              f"{run.eval_solved_count:>3}/{run.eval_puzzle_count:<3} "
              f"{run.eval_total_expanded_nodes:>9}")
    for row in manual_rows:
        print(f"{row['name']:>28} {row['eval_fitness']:>9.0f} "
              f"{row['solved']:>3}/{row['puzzles']:<3} {row['expanded']:>9}")
    print(f"\ngp expression diversity: {summary.gp_expression_diversity:.2f}")
    print(f"gep expression diversity: {summary.gep_expression_diversity:.2f}")
    print(f"outputs: {output_dir}/gp_gep_hard_runs.csv, "
          f"{summary_csv}, {output_dir}/gp_gep_hard_summary.txt")
    sys.exit(0)


if __name__ == "__main__":
    main()
