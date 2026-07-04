"""Tests for the random-search sanity baseline and its CLI."""

import subprocess
import sys
from pathlib import Path

import pytest

from src.ackley.csv_io import CSV_HEADER, write_ackley_results
from src.ackley.random_search import run_random_search
from src.ackley.result import AckleyRunResult

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_run_returns_result_dataclass():
    result = run_random_search(iterations=50, seed=1)
    assert isinstance(result, AckleyRunResult)
    assert result.algorithm == "random_search"
    assert result.dimension == 10
    assert result.iterations == 50
    assert len(result.best_x) == 10
    assert result.best_value > 0.0  # random search will not hit the exact optimum


def test_same_seed_is_reproducible():
    first = run_random_search(iterations=100, seed=42)
    second = run_random_search(iterations=100, seed=42)
    assert first.best_value == pytest.approx(second.best_value)
    assert first.best_x == second.best_x


def test_different_seed_changes_result():
    a = run_random_search(iterations=100, seed=1)
    b = run_random_search(iterations=100, seed=2)
    # not a strict guarantee in theory, but for continuous sampling
    # two seeds giving the identical best point would be astonishing
    assert a.best_x != b.best_x


def test_distance_from_origin_non_negative():
    result = run_random_search(iterations=50, seed=3)
    assert result.distance_from_origin >= 0.0


def test_write_ackley_results(tmp_path):
    result = run_random_search(iterations=20, seed=5)
    csv_path = tmp_path / "sub" / "ackley.csv"
    write_ackley_results([result], csv_path)
    lines = csv_path.read_text().splitlines()
    assert lines[0] == ",".join(CSV_HEADER)
    assert len(lines) == 2
    assert lines[1].startswith("random_search,10,5,20,")


# ---------- CLI ----------

def run_cli(*extra_args):
    script = REPO_ROOT / "scripts" / "run_ackley_random_search.py"
    return subprocess.run(
        [sys.executable, str(script), *extra_args],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )


def test_cli_small_run():
    proc = run_cli("--iterations", "50", "--seed", "42")
    assert proc.returncode == 0
    assert "algorithm: random_search" in proc.stdout
    assert "best_value:" in proc.stdout
    assert "distance_from_origin:" in proc.stdout


def test_cli_writes_csv(tmp_path):
    csv_path = tmp_path / "ackley_random_search.csv"
    proc = run_cli("--iterations", "50", "--seed", "42", "--output", str(csv_path))
    assert proc.returncode == 0
    assert csv_path.exists()
    assert csv_path.read_text().splitlines()[0] == ",".join(CSV_HEADER)
