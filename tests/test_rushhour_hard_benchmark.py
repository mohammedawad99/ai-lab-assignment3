"""Tests for the hard Rush Hour benchmark, puzzles and manual heuristics."""

import subprocess
import sys
from pathlib import Path

from src.rushhour.astar import solve_astar
from src.rushhour.heuristics import (
    BASELINE_HEURISTICS,
    blocker_depth_heuristic,
    blocking_heuristic,
    zero_heuristic,
)
from src.rushhour.parser import find_vehicles, parse_board_string
from src.rushhour.puzzle_sets import load_puzzle_set

REPO_ROOT = Path(__file__).resolve().parents[1]
HARD_FILE = REPO_ROOT / "examples" / "rushhour_hard_eval.txt"


def hard_puzzles():
    return load_puzzle_set(HARD_FILE)


def test_hard_file_has_enough_valid_puzzles():
    puzzles = hard_puzzles()
    assert len(puzzles) >= 10
    for p in puzzles:
        find_vehicles(p)  # raises on invalid vehicles
        assert "X" in p.board


def test_hard_puzzles_are_solvable_and_not_all_trivial():
    puzzles = hard_puzzles()
    expansions = []
    for p in puzzles:
        result = solve_astar(p, heuristic=blocker_depth_heuristic,
                             max_nodes=200000, timeout_sec=30.0)
        assert result.solved, "hard puzzle must stay solvable"
        assert result.cost >= 4  # no one/two-move puzzles
        expansions.append(result.expanded_nodes)
    # several puzzles must need real search even with the best manual guide
    assert sum(1 for e in expansions if e >= 300) >= 4, expansions


def test_manual_heuristics_numeric_and_pure():
    puzzles = hard_puzzles()
    for name, func in BASELINE_HEURISTICS.items():
        for p in puzzles:
            board_before = p.board
            value = func(p)
            assert isinstance(value, (int, float)), name
            assert value >= 0
            assert p.board == board_before  # no mutation


def test_blocker_depth_orders_sensibly():
    solved = parse_board_string("......" "......" "XX...." "......" "......" "......")
    one_blocker = parse_board_string("......" "..A..." "XXA..." "......" "......" "......")
    assert blocking_heuristic(solved) == 0
    assert blocking_heuristic(one_blocker) == 1
    assert blocker_depth_heuristic(one_blocker) >= blocking_heuristic(one_blocker)
    assert zero_heuristic(one_blocker) == 0


def run_script(script, *extra):
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / script), *extra],
        capture_output=True, text=True, cwd=REPO_ROOT)


def test_scripts_support_help():
    for script in ("evaluate_rushhour_heuristics.py", "run_gp_gep_hard_benchmark.py"):
        proc = run_script(script, "--help")
        assert proc.returncode == 0
        assert "--puzzles" in proc.stdout


def test_hard_benchmark_smoke(tmp_path):
    proc = run_script(
        "run_gp_gep_hard_benchmark.py",
        "--puzzles", str(HARD_FILE), "--seeds", "7",
        "--generations", "1", "--population-size", "4",
        "--train-hard-count", "1",
        "--max-nodes-per-puzzle", "800", "--max-time-per-puzzle", "0.3",
        "--max-total-time", "3", "--output-dir", str(tmp_path))
    assert proc.returncode == 0, proc.stderr
    assert (tmp_path / "gp_gep_hard_runs.csv").exists()
    assert (tmp_path / "gp_gep_hard_summary.csv").exists()
