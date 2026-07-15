"""Tests for the discretized LDS-style Ackley variant."""

import math
import subprocess
import sys
from pathlib import Path

import pytest

from src.ackley.bnb_lds import (
    bin_center,
    make_bins,
    point_from_bin_choices,
    run_ackley_bnb_lds,
    sorted_bin_indices_by_center,
)
from src.ackley.function import default_bounds, is_inside_bounds
from src.ackley.result import AckleyRunResult

REPO_ROOT = Path(__file__).resolve().parents[1]

SMALL = dict(dimension=4, bins_per_dimension=5, max_discrepancy=2,
             max_nodes=5000, timeout_sec=5.0)


def test_make_bins():
    bins = make_bins(-32.768, 32.768, 5)
    assert len(bins) == 5
    assert bins[0][0] == pytest.approx(-32.768)
    assert bins[-1][1] == pytest.approx(32.768)
    for lo, hi in bins:
        assert -32.768 <= bin_center((lo, hi)) <= 32.768


def test_sorted_bin_indices_by_center():
    bins = make_bins(-32.768, 32.768, 5)
    order = sorted_bin_indices_by_center(bins)
    # with 5 bins the middle bin is centered at 0 and must come first
    assert order[0] == 2
    assert bin_center(bins[order[0]]) == pytest.approx(0.0)


def test_point_from_bin_choices():
    bins = make_bins(-32.768, 32.768, 5)
    point = point_from_bin_choices([2, 2, 0, 4], bins)
    assert len(point) == 4
    assert point[0] == pytest.approx(0.0)


def test_bnb_lds_returns_result():
    result = run_ackley_bnb_lds(seed=1, **SMALL)
    assert isinstance(result, AckleyRunResult)
    assert result.algorithm == "ackley_bnb_lds"
    assert is_inside_bounds(result.best_x, default_bounds(4))
    assert math.isfinite(result.best_value)
    assert result.best_value >= 0.0


def test_bnb_lds_reproducible():
    a = run_ackley_bnb_lds(seed=42, **SMALL)
    b = run_ackley_bnb_lds(seed=42, **SMALL)
    assert a.best_value == pytest.approx(b.best_value)
    assert a.best_x == b.best_x


def test_bnb_lds_max_nodes_safety():
    result = run_ackley_bnb_lds(dimension=4, bins_per_dimension=5,
                                max_discrepancy=2, max_nodes=2, seed=42,
                                timeout_sec=5.0)
    # too few nodes for a full point: the random fallback is still valid
    assert result.iterations <= 2
    assert is_inside_bounds(result.best_x, default_bounds(4))
    assert math.isfinite(result.best_value)


# ---------- CLI ----------

def run_cli(*extra_args):
    script = REPO_ROOT / "scripts" / "run_ackley_bnb_lds.py"
    return subprocess.run(
        [sys.executable, str(script), *extra_args],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )


def test_cli_runs():
    proc = run_cli("--dimension", "4", "--bins-per-dimension", "5",
                   "--max-discrepancy", "2", "--max-nodes", "5000",
                   "--seed", "42", "--timeout", "5")
    assert proc.returncode == 0
    assert "algorithm: ackley_bnb_lds" in proc.stdout
    assert "best_value:" in proc.stdout


def test_cli_writes_csv(tmp_path):
    csv_path = tmp_path / "ackley_bnb_lds.csv"
    proc = run_cli("--dimension", "4", "--bins-per-dimension", "5",
                   "--max-discrepancy", "2", "--max-nodes", "5000",
                   "--seed", "42", "--output", str(csv_path))
    assert proc.returncode == 0
    assert csv_path.exists()
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert lines[1].startswith("ackley_bnb_lds,")
