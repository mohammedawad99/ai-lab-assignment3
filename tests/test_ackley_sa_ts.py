"""Tests for the Ackley SA and Tabu Search foundations."""

import math
import subprocess
import sys
from pathlib import Path

import pytest

from src.ackley.function import default_bounds, is_inside_bounds
from src.ackley.result import AckleyRunResult
from src.ackley.simulated_annealing import run_ackley_simulated_annealing
from src.ackley.tabu_search import run_ackley_tabu_search

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_sa_returns_result():
    result = run_ackley_simulated_annealing(iterations=100, seed=1)
    assert isinstance(result, AckleyRunResult)
    assert result.algorithm == "ackley_sa"
    assert is_inside_bounds(result.best_x, default_bounds(10))
    assert math.isfinite(result.best_value)
    assert result.best_value >= 0.0


def test_tabu_returns_result():
    result = run_ackley_tabu_search(iterations=50, seed=1)
    assert isinstance(result, AckleyRunResult)
    assert result.algorithm == "ackley_tabu"
    assert is_inside_bounds(result.best_x, default_bounds(10))
    assert math.isfinite(result.best_value)
    assert result.best_value >= 0.0


def test_sa_reproducible():
    a = run_ackley_simulated_annealing(iterations=100, seed=42)
    b = run_ackley_simulated_annealing(iterations=100, seed=42)
    assert a.best_value == pytest.approx(b.best_value)
    assert a.best_x == b.best_x


def test_tabu_reproducible():
    a = run_ackley_tabu_search(iterations=50, seed=42)
    b = run_ackley_tabu_search(iterations=50, seed=42)
    assert a.best_value == pytest.approx(b.best_value)
    assert a.best_x == b.best_x


# ---------- CLI ----------

def run_cli(*extra_args):
    script = REPO_ROOT / "scripts" / "run_ackley_sa_ts.py"
    return subprocess.run(
        [sys.executable, str(script), *extra_args],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )


def test_cli_sa():
    proc = run_cli("--algorithm", "sa", "--iterations", "50", "--seed", "42")
    assert proc.returncode == 0
    assert "algorithm: ackley_sa" in proc.stdout


def test_cli_tabu():
    proc = run_cli("--algorithm", "tabu", "--iterations", "30", "--seed", "42")
    assert proc.returncode == 0
    assert "algorithm: ackley_tabu" in proc.stdout


def test_cli_both_writes_csv(tmp_path):
    csv_path = tmp_path / "ackley_sa_ts.csv"
    proc = run_cli("--algorithm", "both", "--iterations", "50", "--seed", "42",
                   "--output", str(csv_path))
    assert proc.returncode == 0
    assert "algorithm: ackley_sa" in proc.stdout
    assert "algorithm: ackley_tabu" in proc.stdout
    assert csv_path.exists()
    lines = csv_path.read_text().splitlines()
    assert len(lines) == 3  # header + one row per algorithm
