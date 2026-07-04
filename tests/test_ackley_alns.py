"""Tests for the ALNS-style Ackley variant."""

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.ackley.alns import (
    gaussian_repair,
    random_dimension_destroy,
    random_repair,
    run_ackley_alns,
    toward_zero_repair,
    worst_dimension_destroy,
)
from src.ackley.function import default_bounds, is_inside_bounds
from src.ackley.result import AckleyRunResult

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_destroy_operators_return_valid_indices():
    rng = np.random.default_rng(0)
    current = [1.0, -20.0, 3.0, 0.5, 30.0, -7.0, 2.0, 0.1, -0.3, 15.0]
    for destroy in (random_dimension_destroy, worst_dimension_destroy):
        for _ in range(20):
            indices = destroy(current, rng)
            assert indices == sorted(set(indices))
            assert all(0 <= i < len(current) for i in indices)
            assert len(indices) >= 1


def test_repair_operators_stay_in_bounds():
    rng = np.random.default_rng(1)
    bounds = default_bounds(10)
    current = [32.0, -32.0] * 5  # near the bounds on purpose
    destroyed = [0, 1, 5, 9]
    for _ in range(20):
        assert is_inside_bounds(random_repair(current, destroyed, bounds, rng), bounds)
        assert is_inside_bounds(
            gaussian_repair(current, destroyed, bounds, rng, mutation_scale=5.0), bounds
        )
        assert is_inside_bounds(toward_zero_repair(current, destroyed, bounds, rng), bounds)


def test_alns_returns_result():
    result = run_ackley_alns(iterations=20, seed=1, timeout_sec=5.0)
    assert isinstance(result, AckleyRunResult)
    assert result.algorithm == "ackley_alns"
    assert is_inside_bounds(result.best_x, default_bounds(10))
    assert math.isfinite(result.best_value)
    assert result.best_value >= 0.0


def test_alns_reproducible():
    a = run_ackley_alns(iterations=20, seed=42, timeout_sec=5.0)
    b = run_ackley_alns(iterations=20, seed=42, timeout_sec=5.0)
    assert a.best_value == pytest.approx(b.best_value)
    assert a.best_x == b.best_x


# ---------- CLI ----------

def run_cli(*extra_args):
    script = REPO_ROOT / "scripts" / "run_ackley_alns.py"
    return subprocess.run(
        [sys.executable, str(script), *extra_args],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )


def test_cli_runs():
    proc = run_cli("--iterations", "20", "--seed", "42", "--timeout", "5")
    assert proc.returncode == 0
    assert "algorithm: ackley_alns" in proc.stdout
    assert "best_value:" in proc.stdout


def test_cli_writes_csv(tmp_path):
    csv_path = tmp_path / "ackley_alns.csv"
    proc = run_cli("--iterations", "20", "--seed", "42", "--output", str(csv_path))
    assert proc.returncode == 0
    assert csv_path.exists()
    lines = csv_path.read_text().splitlines()
    assert len(lines) == 2
    assert lines[1].startswith("ackley_alns,")
