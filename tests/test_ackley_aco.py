"""Tests for the discretized ACO-style Ackley variant."""

import math
import subprocess
import sys
from pathlib import Path

import pytest

from src.ackley.aco import run_ackley_aco
from src.ackley.function import default_bounds, is_inside_bounds
from src.ackley.result import AckleyRunResult

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_aco_returns_result():
    result = run_ackley_aco(iterations=5, ants=5, bins_per_dimension=10, seed=1,
                            timeout_sec=5.0)
    assert isinstance(result, AckleyRunResult)
    assert result.algorithm == "ackley_aco"
    assert is_inside_bounds(result.best_x, default_bounds(10))
    assert math.isfinite(result.best_value)
    assert result.best_value >= 0.0


def test_aco_reproducible():
    a = run_ackley_aco(iterations=5, ants=5, bins_per_dimension=10, seed=42,
                       timeout_sec=5.0)
    b = run_ackley_aco(iterations=5, ants=5, bins_per_dimension=10, seed=42,
                       timeout_sec=5.0)
    assert a.best_value == pytest.approx(b.best_value)
    assert a.best_x == b.best_x


# ---------- CLI ----------

def run_cli(*extra_args):
    script = REPO_ROOT / "scripts" / "run_ackley_aco.py"
    return subprocess.run(
        [sys.executable, str(script), *extra_args],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )


def test_cli_runs():
    proc = run_cli("--iterations", "5", "--ants", "5", "--bins-per-dimension", "10",
                   "--seed", "42", "--timeout", "5")
    assert proc.returncode == 0
    assert "algorithm: ackley_aco" in proc.stdout
    assert "best_value:" in proc.stdout


def test_cli_writes_csv(tmp_path):
    csv_path = tmp_path / "ackley_aco.csv"
    proc = run_cli("--iterations", "5", "--ants", "5", "--seed", "42",
                   "--output", str(csv_path))
    assert proc.returncode == 0
    assert csv_path.exists()
    lines = csv_path.read_text().splitlines()
    assert len(lines) == 2
    assert lines[1].startswith("ackley_aco,")
