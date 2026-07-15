"""Tests for the safe Rush Hour heuristic evaluator."""

import subprocess
import sys
from pathlib import Path

from src.rushhour.evaluator import (
    CSV_HEADER,
    evaluate_heuristic,
    fitness_from_evaluation,
    write_evaluation_csv,
)
from src.rushhour.heuristics import blocking_distance_heuristic, zero_heuristic
from src.rushhour.model import RushHourState
from src.rushhour.puzzle_sets import load_puzzle_set

REPO_ROOT = Path(__file__).resolve().parents[1]
PUZZLES_FILE = REPO_ROOT / "examples" / "rushhour_eval_puzzles.txt"


def load_puzzles():
    return load_puzzle_set(PUZZLES_FILE)


def test_load_puzzle_set():
    puzzles = load_puzzles()
    assert len(puzzles) == 3
    assert all(isinstance(p, RushHourState) for p in puzzles)
    assert all(len(p.board) == 36 for p in puzzles)


def test_evaluate_with_zero_heuristic():
    evaluation = evaluate_heuristic("zero", zero_heuristic, load_puzzles())
    assert evaluation.puzzle_count == 3
    assert evaluation.solved_count >= 1
    assert len(evaluation.puzzle_results) == 3


def test_evaluate_with_blocking_distance_heuristic():
    evaluation = evaluate_heuristic(
        "blocking_distance", blocking_distance_heuristic, load_puzzles()
    )
    assert evaluation.solved_count >= 1
    assert 0.0 <= evaluation.solved_rate <= 1.0


def test_max_nodes_cap_is_respected():
    evaluation = evaluate_heuristic(
        "capped", blocking_distance_heuristic, load_puzzles(),
        max_nodes_per_puzzle=1,
    )
    assert all(r.expanded_nodes <= 1 for r in evaluation.puzzle_results)
    # puzzle 3 needs more than one expansion, so the cap must trigger
    assert evaluation.max_nodes_count >= 1


def test_total_time_budget_does_not_hang():
    evaluation = evaluate_heuristic(
        "no_time", blocking_distance_heuristic, load_puzzles(),
        max_total_time_sec=0.0,
    )
    # no A* runs at all: every puzzle is counted as a timeout
    assert evaluation.solved_count == 0
    assert evaluation.timeout_count == 3
    assert evaluation.total_expanded_nodes == 0


def test_heuristic_error_is_caught():
    def broken_heuristic(state):
        raise RuntimeError("bad heuristic")

    evaluation = evaluate_heuristic("broken", broken_heuristic, load_puzzles())
    assert evaluation.solved_count == 0
    assert all(r.stopped_reason == "heuristic_error" for r in evaluation.puzzle_results)


def test_fitness_is_numeric():
    evaluation = evaluate_heuristic("zero", zero_heuristic, load_puzzles())
    fitness = fitness_from_evaluation(evaluation)
    assert isinstance(fitness, float)
    # solving at least one puzzle should dominate the small node penalties here
    assert fitness > 0


def test_write_evaluation_csv(tmp_path):
    evaluation = evaluate_heuristic("zero", zero_heuristic, load_puzzles())
    csv_path = tmp_path / "sub" / "eval.csv"
    write_evaluation_csv([evaluation], csv_path)
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == ",".join(CSV_HEADER)
    assert len(lines) == 2
    assert lines[1].startswith("zero,3,")


# ---------- CLI ----------

def run_cli(*extra_args):
    script = REPO_ROOT / "scripts" / "evaluate_rushhour_heuristics.py"
    return subprocess.run(
        [sys.executable, str(script), "--puzzles", str(PUZZLES_FILE), *extra_args],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )


def test_cli_runs():
    proc = run_cli()
    assert proc.returncode == 0
    assert "loaded 3 puzzles" in proc.stdout
    assert "blocking_distance:" in proc.stdout


def test_cli_writes_csv(tmp_path):
    csv_path = tmp_path / "rushhour_heuristics.csv"
    proc = run_cli("--output", str(csv_path))
    assert proc.returncode == 0
    assert csv_path.exists()
    assert csv_path.read_text(encoding="utf-8").splitlines()[0] == ",".join(CSV_HEADER)


def test_cli_unknown_heuristic_fails():
    proc = run_cli("--heuristics", "unknown_name")
    assert proc.returncode == 1
    assert "unknown heuristic" in proc.stderr
