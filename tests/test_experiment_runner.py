"""Tests for the unified CVRP/Ackley experiment runner."""

import subprocess
import sys
from pathlib import Path

import pytest

from src.experiments.ackley_experiments import ACKLEY_FIELDNAMES, run_ackley_experiments
from src.experiments.csv_utils import append_dict_rows, write_dict_rows
from src.experiments.cvrp_experiments import CVRP_FIELDNAMES, run_cvrp_experiments

REPO_ROOT = Path(__file__).resolve().parents[1]
TINY_INSTANCE = REPO_ROOT / "examples" / "tiny_cvrp.vrp"


# ---------- csv utils ----------

def test_write_and_append_dict_rows(tmp_path):
    path = tmp_path / "sub" / "rows.csv"
    fieldnames = ["a", "b"]
    write_dict_rows(path, [{"a": 1, "b": 2}], fieldnames)
    append_dict_rows(path, [{"a": 3, "b": 4}], fieldnames)
    lines = path.read_text().splitlines()
    assert lines == ["a,b", "1,2", "3,4"]  # single header only


def test_write_dict_rows_empty_still_has_header(tmp_path):
    path = tmp_path / "empty.csv"
    write_dict_rows(path, [], ["x", "y"])
    assert path.read_text().splitlines() == ["x,y"]


# ---------- cvrp runner ----------

def test_run_cvrp_experiments(tmp_path):
    rows = run_cvrp_experiments([TINY_INSTANCE], ["sa", "tabu"], [42],
                                budget=10, timeout_sec=5.0, output_dir=tmp_path)
    assert len(rows) == 2
    for row in rows:
        assert set(CVRP_FIELDNAMES) <= set(row.keys())
        assert row["part"] == "cvrp"
        assert row["instance"] == "tiny-cvrp"
        assert row["feasible"] is True
        assert float(row["best_cost"]) == pytest.approx(80.64, abs=0.05)
        assert row["solution_path"]
        assert Path(row["solution_path"]).exists()
    solution_files = list((tmp_path / "cvrp_solutions").glob("*.txt"))
    assert len(solution_files) == 2


def test_run_cvrp_experiments_with_baseline_and_bks():
    rows = run_cvrp_experiments([TINY_INSTANCE], ["baseline"], [42],
                                budget=5, timeout_sec=5.0,
                                bks_by_instance={"tiny-cvrp": 80.64})
    assert len(rows) == 1
    row = rows[0]
    assert row["algorithm"] == "baseline"
    assert row["gap_percent"] != ""
    assert abs(float(row["gap_percent"])) < 0.1


# ---------- ackley runner ----------

def test_run_ackley_experiments():
    rows = run_ackley_experiments(["sa", "tabu"], [42], budget=5,
                                  timeout_sec=5.0, dimension=4)
    assert len(rows) == 2
    for row in rows:
        assert set(ACKLEY_FIELDNAMES) <= set(row.keys())
        assert row["part"] == "ackley"
        assert row["dimension"] == 4
        assert float(row["best_value"]) >= 0.0 or row["errors"]
        assert row["best_x"]


# ---------- CLI ----------

def run_cli(*extra_args):
    script = REPO_ROOT / "scripts" / "run_experiments.py"
    return subprocess.run(
        [sys.executable, str(script), *extra_args],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )


def test_cli_cvrp(tmp_path):
    output = tmp_path / "cvrp_smoke.csv"
    proc = run_cli("--part", "cvrp", "--instances", str(TINY_INSTANCE),
                   "--algorithms", "sa", "--seeds", "42", "--budget", "5",
                   "--timeout", "5", "--output", str(output))
    assert proc.returncode == 0
    assert output.exists()
    assert "best cvrp on tiny-cvrp" in proc.stdout


def test_cli_ackley(tmp_path):
    output = tmp_path / "ackley_smoke.csv"
    proc = run_cli("--part", "ackley", "--algorithms", "sa", "--seeds", "42",
                   "--budget", "5", "--timeout", "5", "--dimension", "4",
                   "--output", str(output))
    assert proc.returncode == 0
    assert output.exists()
    assert "best ackley:" in proc.stdout


def test_cli_both(tmp_path):
    output_dir = tmp_path / "smoke"
    proc = run_cli("--part", "both", "--instances", str(TINY_INSTANCE),
                   "--algorithms", "sa", "--seeds", "42", "--budget", "5",
                   "--timeout", "5", "--dimension", "4",
                   "--output-dir", str(output_dir))
    assert proc.returncode == 0
    assert (output_dir / "cvrp_results.csv").exists()
    assert (output_dir / "ackley_results.csv").exists()
    assert list((output_dir / "cvrp_solutions").glob("*.txt"))


def test_cli_unknown_algorithm(tmp_path):
    proc = run_cli("--part", "ackley", "--algorithms", "unknown_name",
                   "--seeds", "42", "--budget", "5", "--timeout", "5",
                   "--dimension", "4", "--output", str(tmp_path / "bad.csv"))
    assert proc.returncode == 1
    assert "unknown algorithm" in proc.stderr
