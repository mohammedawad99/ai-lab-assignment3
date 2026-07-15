"""Tests for the Rush Hour A* search and the solver CLI."""

import subprocess
import sys
from pathlib import Path

from src.rushhour.astar import solve_astar
from src.rushhour.heuristics import zero_heuristic
from src.rushhour.parser import parse_board_string

REPO_ROOT = Path(__file__).resolve().parents[1]
TINY_PUZZLE = REPO_ROOT / "examples" / "rushhour_tiny.txt"

# column 5 is completely filled by P and Q, so X can never reach the exit
UNSOLVABLE_BOARD = (
    ".....P"
    ".....P"
    "XX...P"
    ".....Q"
    ".....Q"
    ".....Q"
)


def load_tiny_state():
    return parse_board_string(TINY_PUZZLE.read_text(encoding="utf-8"))


def test_astar_solves_tiny_puzzle():
    result = solve_astar(load_tiny_state())
    assert result.solved
    assert result.cost == 1
    assert result.stopped_reason == "solved"
    assert len(result.moves) == 1
    assert result.moves[0].car == "A"
    assert result.expanded_nodes > 0
    assert result.generated_nodes > 0


def test_astar_with_zero_heuristic():
    result = solve_astar(load_tiny_state(), heuristic=zero_heuristic)
    assert result.solved
    assert result.cost == 1


def test_unsolvable_board_reports_no_solution():
    result = solve_astar(parse_board_string(UNSOLVABLE_BOARD))
    assert not result.solved
    assert result.stopped_reason == "no_solution"
    assert result.moves == []


def test_max_nodes_cap_stops_safely():
    result = solve_astar(parse_board_string(UNSOLVABLE_BOARD), max_nodes=1)
    assert not result.solved
    assert result.stopped_reason == "max_nodes"
    assert result.expanded_nodes <= 1


def test_cli_solves_tiny_puzzle():
    script = REPO_ROOT / "scripts" / "solve_rushhour.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--puzzle", str(TINY_PUZZLE)],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert proc.returncode == 0
    assert "solved: yes" in proc.stdout
    assert "cost: 1" in proc.stdout
    assert "stopped_reason: solved" in proc.stdout
