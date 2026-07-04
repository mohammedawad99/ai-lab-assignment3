"""Tests for the Ackley GA Island Model."""

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.ackley.function import default_bounds, is_inside_bounds
from src.ackley.ga_island import (
    blend_crossover,
    gaussian_mutation,
    migrate_ackley_ring,
    random_vector,
    run_ackley_ga_island,
)
from src.ackley.result import AckleyRunResult

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_ga_island_returns_result():
    result = run_ackley_ga_island(generations=5, population_size=8, islands=2,
                                  seed=1, timeout_sec=5.0)
    assert isinstance(result, AckleyRunResult)
    assert result.algorithm == "ackley_ga_island"
    assert is_inside_bounds(result.best_x, default_bounds(10))
    assert math.isfinite(result.best_value)
    assert result.best_value >= 0.0


def test_ga_island_reproducible():
    a = run_ackley_ga_island(generations=5, population_size=8, islands=2,
                             seed=42, timeout_sec=5.0)
    b = run_ackley_ga_island(generations=5, population_size=8, islands=2,
                             seed=42, timeout_sec=5.0)
    assert a.best_value == pytest.approx(b.best_value)
    assert a.best_x == b.best_x


def test_blend_crossover_within_widened_range():
    rng = np.random.default_rng(0)
    p1, p2 = [0.0, -10.0, 5.0], [10.0, -20.0, 5.0]
    for _ in range(20):
        child = blend_crossover(p1, p2, rng, alpha=0.5)
        assert len(child) == 3
        for value, a, b in zip(child, p1, p2):
            low, high = min(a, b), max(a, b)
            spread = high - low
            assert low - 0.5 * spread - 1e-9 <= value <= high + 0.5 * spread + 1e-9


def test_gaussian_mutation_stays_in_bounds():
    rng = np.random.default_rng(1)
    bounds = default_bounds(10)
    vector = [32.0] * 10  # close to the upper bound
    for _ in range(20):
        mutated = gaussian_mutation(vector, bounds, rng, mutation_rate=1.0,
                                    mutation_scale=5.0)
        assert is_inside_bounds(mutated, bounds)


def test_migrate_ackley_ring_keeps_sizes():
    rng = np.random.default_rng(2)
    bounds = default_bounds(5)
    islands = [[random_vector(5, bounds, rng) for _ in range(6)] for _ in range(3)]
    migrated = migrate_ackley_ring(islands, migrants=2)
    assert len(migrated) == 3
    assert all(len(island) == 6 for island in migrated)


# ---------- CLI ----------

def run_cli(*extra_args):
    script = REPO_ROOT / "scripts" / "run_ackley_ga_island.py"
    return subprocess.run(
        [sys.executable, str(script), *extra_args],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )


def test_cli_runs():
    proc = run_cli("--generations", "5", "--population-size", "8",
                   "--islands", "2", "--seed", "42", "--timeout", "5")
    assert proc.returncode == 0
    assert "algorithm: ackley_ga_island" in proc.stdout
    assert "best_value:" in proc.stdout


def test_cli_writes_csv(tmp_path):
    csv_path = tmp_path / "ackley_ga_island.csv"
    proc = run_cli("--generations", "5", "--population-size", "8",
                   "--islands", "2", "--seed", "42", "--output", str(csv_path))
    assert proc.returncode == 0
    assert csv_path.exists()
    lines = csv_path.read_text().splitlines()
    assert len(lines) == 2
    assert lines[1].startswith("ackley_ga_island,")
