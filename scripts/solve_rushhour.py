"""Solve one Rush Hour puzzle file with A*.

Usage:
    python scripts/solve_rushhour.py --puzzle examples/rushhour_tiny.txt \
        [--heuristic zero|blocking|blocking_distance] [--max-nodes N] [--timeout SEC]
"""

import argparse
import sys
from pathlib import Path

# make "src" importable when the script is run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rushhour.astar import solve_astar
from src.rushhour.heuristics import BASELINE_HEURISTICS
from src.rushhour.parser import parse_board_string


def main():
    parser = argparse.ArgumentParser(description="Solve a Rush Hour puzzle with A*.")
    parser.add_argument("--puzzle", required=True, help="path to a puzzle file (6x6 board)")
    parser.add_argument("--heuristic", choices=sorted(BASELINE_HEURISTICS),
                        default="blocking_distance", help="heuristic to use")
    parser.add_argument("--max-nodes", type=int, default=100000,
                        help="stop after expanding this many nodes")
    parser.add_argument("--timeout", type=float, default=10.0,
                        help="time limit in seconds")
    args = parser.parse_args()

    with open(args.puzzle) as f:
        state = parse_board_string(f.read())

    result = solve_astar(state, heuristic=BASELINE_HEURISTICS[args.heuristic],
                         max_nodes=args.max_nodes, timeout_sec=args.timeout)

    print(f"solved: {'yes' if result.solved else 'no'}")
    print(f"cost: {result.cost}")
    print(f"expanded_nodes: {result.expanded_nodes}")
    print(f"generated_nodes: {result.generated_nodes}")
    print(f"elapsed_time: {result.elapsed_time:.4f}")
    print(f"cpu_time: {result.cpu_time:.4f}")
    print(f"stopped_reason: {result.stopped_reason}")
    print("moves: " + " ".join(str(move) for move in result.moves))

    sys.exit(0 if result.solved else 1)


if __name__ == "__main__":
    main()
