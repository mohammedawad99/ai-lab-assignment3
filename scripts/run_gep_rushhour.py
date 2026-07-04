"""Evolve Rush Hour heuristics with GEP (linear Karva genomes).

Usage:
    python scripts/run_gep_rushhour.py --puzzles examples/rushhour_gep_train.txt
"""

import argparse
import csv
import sys
from pathlib import Path

# make "src" importable when the script is run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gep.evolution import run_gep
from src.rushhour.puzzle_sets import load_puzzle_set

HISTORY_HEADER = ["generation", "best_fitness", "best_solved_count",
                  "best_total_expanded_nodes", "best_total_cost",
                  "best_expression", "best_genome"]


def main():
    parser = argparse.ArgumentParser(description="GEP for Rush Hour heuristics.")
    parser.add_argument("--puzzles", required=True, help="path to a puzzle set file")
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--population-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--head-length", type=int, default=6)
    parser.add_argument("--crossover-rate", type=float, default=0.8)
    parser.add_argument("--mutation-rate", type=float, default=0.05)
    parser.add_argument("--max-nodes-per-puzzle", type=int, default=10000)
    parser.add_argument("--max-time-per-puzzle", type=float, default=1.0)
    parser.add_argument("--max-total-time", type=float, default=10.0)
    parser.add_argument("--output", help="optional path for the history CSV")
    parser.add_argument("--best-output", help="optional path for the best expression file")
    args = parser.parse_args()

    puzzles = load_puzzle_set(args.puzzles)
    print(f"loaded {len(puzzles)} puzzles from {args.puzzles}")

    result = run_gep(
        puzzles,
        generations=args.generations,
        population_size=args.population_size,
        seed=args.seed,
        head_length=args.head_length,
        crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate,
        max_nodes_per_puzzle=args.max_nodes_per_puzzle,
        max_time_per_puzzle_sec=args.max_time_per_puzzle,
        max_total_time_sec=args.max_total_time,
    )

    best = result.best
    print(f"algorithm: {result.algorithm}")
    print(f"seed: {result.seed}")
    print(f"generations: {result.generations}")
    print(f"population_size: {result.population_size}")
    print(f"head_length: {result.head_length}")
    print(f"best_fitness: {best.fitness:.0f}")
    print(f"best_expression: {best.expression}")
    print(f"best_genome: {best.genome_string}")
    print(f"best_solved_count: {best.solved_count}/{best.puzzle_count}")
    print(f"best_total_expanded_nodes: {best.total_expanded_nodes}")
    print(f"best_total_cost: {best.total_cost}")
    print(f"elapsed_time: {result.elapsed_time:.4f}")
    print(f"cpu_time: {result.cpu_time:.4f}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(HISTORY_HEADER)
            for row in result.history:
                writer.writerow([row[key] for key in HISTORY_HEADER])
        print(f"history csv written to: {output_path}")

    if args.best_output:
        best_path = Path(args.best_output)
        best_path.parent.mkdir(parents=True, exist_ok=True)
        best_path.write_text(
            f"expression: {best.expression}\n"
            f"genome: {best.genome_string}\n"
            f"fitness: {best.fitness:.0f}\n"
            f"solved: {best.solved_count}/{best.puzzle_count}\n"
            f"expanded_nodes: {best.total_expanded_nodes}\n"
            f"total_cost: {best.total_cost}\n"
        )
        print(f"best expression written to: {best_path}")

    sys.exit(0)


if __name__ == "__main__":
    main()
