"""Compare GP and GEP Rush Hour heuristics on shared puzzle sets.

Usage:
    python scripts/compare_gp_gep_rushhour.py \
        --train-puzzles examples/rushhour_gp_train.txt \
        --eval-puzzles examples/rushhour_gp_gep_eval.txt
"""

import argparse
import sys
from pathlib import Path

# make "src" importable when the script is run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rushhour.gp_gep_comparison import (
    compare_gp_gep,
    write_comparison_summary_txt,
    write_run_summaries_csv,
)
from src.rushhour.puzzle_sets import load_puzzle_set


def main():
    parser = argparse.ArgumentParser(description="Compare GP and GEP on Rush Hour.")
    parser.add_argument("--train-puzzles", required=True)
    parser.add_argument("--eval-puzzles", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--population-size", type=int, default=20)
    parser.add_argument("--gp-max-depth", type=int, default=4)
    parser.add_argument("--gep-head-length", type=int, default=6)
    parser.add_argument("--crossover-rate", type=float, default=0.8)
    parser.add_argument("--gp-mutation-rate", type=float, default=0.2)
    parser.add_argument("--gep-mutation-rate", type=float, default=0.05)
    parser.add_argument("--max-nodes-per-puzzle", type=int, default=10000)
    parser.add_argument("--max-time-per-puzzle", type=float, default=1.0)
    parser.add_argument("--max-total-time", type=float, default=10.0)
    parser.add_argument("--output", help="optional path for the run-level CSV")
    parser.add_argument("--summary-output", help="optional path for the text summary")
    args = parser.parse_args()

    train_puzzles = load_puzzle_set(args.train_puzzles)
    eval_puzzles = load_puzzle_set(args.eval_puzzles)
    print(f"train puzzles: {len(train_puzzles)}")
    print(f"eval puzzles: {len(eval_puzzles)}")
    print(f"seeds: {' '.join(str(s) for s in args.seeds)}")

    summary = compare_gp_gep(
        train_puzzles, eval_puzzles, args.seeds,
        generations=args.generations,
        population_size=args.population_size,
        gp_max_depth=args.gp_max_depth,
        gep_head_length=args.gep_head_length,
        crossover_rate=args.crossover_rate,
        gp_mutation_rate=args.gp_mutation_rate,
        gep_mutation_rate=args.gep_mutation_rate,
        max_nodes_per_puzzle=args.max_nodes_per_puzzle,
        max_time_per_puzzle_sec=args.max_time_per_puzzle,
        max_total_time_sec=args.max_total_time,
    )

    print(f"gp expression diversity: {summary.gp_expression_diversity:.2f}")
    print(f"gep expression diversity: {summary.gep_expression_diversity:.2f}")
    print(f"gep genome diversity: {summary.gep_genome_diversity:.2f}")

    best_gp = summary.best_gp
    print(f"best gp eval fitness: {best_gp.eval_fitness:.0f}")
    print(f"best gp expression: {best_gp.best_expression}")
    print(f"best gp solved: {best_gp.eval_solved_count}/{best_gp.eval_puzzle_count}")
    print(f"best gp eval expanded nodes: {best_gp.eval_total_expanded_nodes}")

    best_gep = summary.best_gep
    print(f"best gep eval fitness: {best_gep.eval_fitness:.0f}")
    print(f"best gep expression: {best_gep.best_expression}")
    print(f"best gep genome: {best_gep.best_genome}")
    print(f"best gep solved: {best_gep.eval_solved_count}/{best_gep.eval_puzzle_count}")
    print(f"best gep eval expanded nodes: {best_gep.eval_total_expanded_nodes}")

    if args.output:
        write_run_summaries_csv(summary, args.output)
        print(f"runs csv written to: {args.output}")
    if args.summary_output:
        write_comparison_summary_txt(summary, args.summary_output)
        print(f"summary written to: {args.summary_output}")

    sys.exit(0)


if __name__ == "__main__":
    main()
