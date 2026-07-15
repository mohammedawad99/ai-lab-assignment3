"""Tests for the GP evolution loop and its CLI."""

import subprocess
import sys
from pathlib import Path

import pytest

from src.gp.evolution import GPIndividualResult, GPRunResult, evaluate_tree, run_gp
from src.gp.tree import GPNode, GPTree
from src.rushhour.puzzle_sets import load_puzzle_set

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = REPO_ROOT / "examples" / "rushhour_gp_train.txt"

SMALL = dict(generations=2, population_size=6, max_nodes_per_puzzle=1000,
             max_time_per_puzzle_sec=0.5, max_total_time_sec=3.0)


def load_train():
    return load_puzzle_set(TRAIN_FILE)


def test_train_file_loads():
    puzzles = load_train()
    assert len(puzzles) == 4
    assert all(len(p.board) == 36 for p in puzzles)


def test_evaluate_tree():
    tree = GPTree(root=GPNode("binary", "+", [
        GPNode("terminal", "blocking"), GPNode("terminal", "distance"),
    ]))
    result = evaluate_tree(tree, load_train(), max_nodes_per_puzzle=1000,
                           max_time_per_puzzle_sec=0.5, max_total_time_sec=3.0)
    assert isinstance(result, GPIndividualResult)
    assert result.puzzle_count == 4
    assert result.solved_count >= 1
    assert result.expression == "(blocking + distance)"
    assert isinstance(result.fitness, float)


def test_evaluate_tree_survives_bad_tree():
    # a tree with a nonsense terminal raises inside the heuristic;
    # the safe evaluator must swallow it
    bad = GPTree(root=GPNode("terminal", "nonsense"))
    result = evaluate_tree(bad, load_train(), max_nodes_per_puzzle=1000,
                           max_time_per_puzzle_sec=0.5, max_total_time_sec=3.0)
    assert result.solved_count == 0
    assert isinstance(result.fitness, float)


def test_run_gp():
    result = run_gp(load_train(), seed=42, **SMALL)
    assert isinstance(result, GPRunResult)
    assert result.algorithm == "gp_rushhour"
    assert isinstance(result.best.fitness, float)
    assert result.best.expression
    assert result.history
    assert result.history[-1]["generation"] == result.generations
    assert "best_expression" in result.history[-1]


def test_run_gp_reproducible():
    a = run_gp(load_train(), seed=7, **SMALL)
    b = run_gp(load_train(), seed=7, **SMALL)
    assert a.best.fitness == pytest.approx(b.best.fitness)
    assert a.best.expression == b.best.expression


# ---------- CLI ----------

def run_cli(*extra_args):
    script = REPO_ROOT / "scripts" / "run_gp_rushhour.py"
    return subprocess.run(
        [sys.executable, str(script), "--puzzles", str(TRAIN_FILE),
         "--generations", "2", "--population-size", "6",
         "--max-nodes-per-puzzle", "1000", "--max-time-per-puzzle", "0.5",
         "--max-total-time", "3", "--seed", "42", *extra_args],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )


def test_cli_runs():
    proc = run_cli()
    assert proc.returncode == 0
    assert "algorithm: gp_rushhour" in proc.stdout
    assert "best_expression:" in proc.stdout


def test_cli_writes_outputs(tmp_path):
    csv_path = tmp_path / "gp_history.csv"
    best_path = tmp_path / "gp_best.txt"
    proc = run_cli("--output", str(csv_path), "--best-output", str(best_path))
    assert proc.returncode == 0
    assert csv_path.exists()
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("generation,best_fitness")
    assert len(lines) == 3  # header + 2 generations
    assert best_path.exists()
    assert "expression:" in best_path.read_text(encoding="utf-8")
