"""Tests for the GP vs GEP comparison runner."""

import subprocess
import sys
from pathlib import Path

from src.rushhour.gp_gep_comparison import (
    CSV_HEADER,
    ComparisonSummary,
    compare_gp_gep,
    expression_diversity,
    genome_diversity,
    write_comparison_summary_txt,
    write_run_summaries_csv,
)
from src.rushhour.puzzle_sets import load_puzzle_set

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = REPO_ROOT / "examples" / "rushhour_gp_train.txt"
EVAL_FILE = REPO_ROOT / "examples" / "rushhour_gp_gep_eval.txt"

SMALL = dict(generations=2, population_size=6, gp_max_depth=3, gep_head_length=5,
             max_nodes_per_puzzle=1000, max_time_per_puzzle_sec=0.5,
             max_total_time_sec=3.0)


def small_comparison(seeds=(7,)):
    train = load_puzzle_set(TRAIN_FILE)
    eval_puzzles = load_puzzle_set(EVAL_FILE)
    return compare_gp_gep(train, eval_puzzles, list(seeds), **SMALL)


def test_eval_file_loads():
    puzzles = load_puzzle_set(EVAL_FILE)
    assert len(puzzles) == 4
    assert all(len(p.board) == 36 for p in puzzles)


def test_expression_diversity():
    assert expression_diversity([]) == 0.0
    assert expression_diversity(["a", "a", "a"]) == 1 / 3
    assert expression_diversity(["a", "b", "c"]) == 1.0


def test_genome_diversity():
    assert genome_diversity([]) == 0.0
    assert genome_diversity([None, None]) == 0.0
    assert genome_diversity(["g1", "g1", None]) == 0.5
    assert genome_diversity(["g1", "g2"]) == 1.0


def test_compare_gp_gep():
    summary = small_comparison()
    assert isinstance(summary, ComparisonSummary)
    assert len(summary.gp_runs) == 1
    assert len(summary.gep_runs) == 1
    assert summary.best_gp.algorithm == "gp_rushhour"
    assert summary.best_gep.algorithm == "gep_rushhour"
    assert 0.0 <= summary.gp_expression_diversity <= 1.0
    assert 0.0 <= summary.gep_expression_diversity <= 1.0
    assert 0.0 <= summary.gep_genome_diversity <= 1.0
    for run in summary.gp_runs + summary.gep_runs:
        assert run.eval_puzzle_count == 4
        assert isinstance(run.eval_fitness, float)
        assert run.eval_total_expanded_nodes >= 0
        assert run.best_expression
    assert summary.best_gp.best_genome is None
    assert summary.best_gep.best_genome


def test_write_outputs(tmp_path):
    summary = small_comparison()
    csv_path = tmp_path / "runs.csv"
    txt_path = tmp_path / "summary.txt"
    write_run_summaries_csv(summary, csv_path)
    write_comparison_summary_txt(summary, txt_path)

    lines = csv_path.read_text().splitlines()
    assert lines[0] == ",".join(CSV_HEADER)
    assert len(lines) == 3  # header + 1 GP run + 1 GEP run

    text = txt_path.read_text()
    assert "gp runs: 1" in text
    assert "best gep genome:" in text
    assert "depend on the small puzzle set" in text


# ---------- CLI ----------

def run_cli(*extra_args):
    script = REPO_ROOT / "scripts" / "compare_gp_gep_rushhour.py"
    return subprocess.run(
        [sys.executable, str(script),
         "--train-puzzles", str(TRAIN_FILE), "--eval-puzzles", str(EVAL_FILE),
         "--seeds", "7", "--generations", "2", "--population-size", "6",
         "--gp-max-depth", "3", "--gep-head-length", "5",
         "--max-nodes-per-puzzle", "1000", "--max-time-per-puzzle", "0.5",
         "--max-total-time", "3", *extra_args],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )


def test_cli_runs():
    proc = run_cli()
    assert proc.returncode == 0
    assert "best gp expression:" in proc.stdout
    assert "best gep genome:" in proc.stdout


def test_cli_writes_outputs(tmp_path):
    csv_path = tmp_path / "comparison_runs.csv"
    txt_path = tmp_path / "comparison_summary.txt"
    proc = run_cli("--output", str(csv_path), "--summary-output", str(txt_path))
    assert proc.returncode == 0
    assert csv_path.exists()
    assert csv_path.read_text().splitlines()[0] == ",".join(CSV_HEADER)
    assert txt_path.exists()
    assert "gep genome diversity" in txt_path.read_text()
