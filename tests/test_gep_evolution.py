"""Tests for the GEP evolution loop and its CLI."""

import subprocess
import sys
from pathlib import Path

import pytest

from src.gep.evolution import GEPIndividualResult, GEPRunResult, evaluate_genome, run_gep
from src.gep.genome import GEPGenome
from src.rushhour.puzzle_sets import load_puzzle_set

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = REPO_ROOT / "examples" / "rushhour_gep_train.txt"

SMALL = dict(generations=2, population_size=6, head_length=5,
             max_nodes_per_puzzle=1000, max_time_per_puzzle_sec=0.5,
             max_total_time_sec=3.0)


def load_train():
    return load_puzzle_set(TRAIN_FILE)


def test_train_file_loads():
    puzzles = load_train()
    assert len(puzzles) == 4
    assert all(len(p.board) == 36 for p in puzzles)


def test_evaluate_genome():
    genome = GEPGenome(genes=["+", "distance", "blocking", "0.0", "0.5", "1.0", "free"],
                       head_length=3)
    result = evaluate_genome(genome, load_train(), max_nodes_per_puzzle=1000,
                             max_time_per_puzzle_sec=0.5, max_total_time_sec=3.0)
    assert isinstance(result, GEPIndividualResult)
    assert result.puzzle_count == 4
    assert result.solved_count >= 1
    assert result.expression == "(distance + blocking)"
    assert result.genome_string.startswith("+ distance blocking")
    assert isinstance(result.fitness, float)


def test_evaluate_genome_weird_but_valid():
    # deep chain of logs over a constant: valid, just a strange heuristic
    genome = GEPGenome(genes=["log", "log", "log", "log", "log", "0.0",
                              "0.0", "0.0", "0.0", "0.0", "0.0"],
                       head_length=5)
    result = evaluate_genome(genome, load_train(), max_nodes_per_puzzle=1000,
                             max_time_per_puzzle_sec=0.5, max_total_time_sec=3.0)
    assert isinstance(result.fitness, float)
    assert result.puzzle_count == 4


def test_run_gep():
    result = run_gep(load_train(), seed=42, **SMALL)
    assert isinstance(result, GEPRunResult)
    assert result.algorithm == "gep_rushhour"
    assert isinstance(result.best.fitness, float)
    assert result.best.expression
    assert result.best.genome_string
    assert result.history
    assert result.history[-1]["generation"] == result.generations
    assert "best_genome" in result.history[-1]


def test_run_gep_reproducible():
    a = run_gep(load_train(), seed=7, **SMALL)
    b = run_gep(load_train(), seed=7, **SMALL)
    assert a.best.fitness == pytest.approx(b.best.fitness)
    assert a.best.expression == b.best.expression
    assert a.best.genome_string == b.best.genome_string


# ---------- CLI ----------

def run_cli(*extra_args):
    script = REPO_ROOT / "scripts" / "run_gep_rushhour.py"
    return subprocess.run(
        [sys.executable, str(script), "--puzzles", str(TRAIN_FILE),
         "--generations", "2", "--population-size", "6", "--head-length", "5",
         "--max-nodes-per-puzzle", "1000", "--max-time-per-puzzle", "0.5",
         "--max-total-time", "3", "--seed", "42", *extra_args],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )


def test_cli_runs():
    proc = run_cli()
    assert proc.returncode == 0
    assert "algorithm: gep_rushhour" in proc.stdout
    assert "best_expression:" in proc.stdout
    assert "best_genome:" in proc.stdout


def test_cli_writes_outputs(tmp_path):
    csv_path = tmp_path / "gep_history.csv"
    best_path = tmp_path / "gep_best.txt"
    proc = run_cli("--output", str(csv_path), "--best-output", str(best_path))
    assert proc.returncode == 0
    assert csv_path.exists()
    lines = csv_path.read_text().splitlines()
    assert lines[0].startswith("generation,best_fitness")
    assert len(lines) == 3  # header + 2 generations
    assert best_path.exists()
    content = best_path.read_text()
    assert "expression:" in content
    assert "genome:" in content
