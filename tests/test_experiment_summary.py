"""Tests for summaries, BKS CLI support and the smoke suite."""

import subprocess
import sys
from pathlib import Path

import pytest

from src.experiments.summary import (
    ACKLEY_SUMMARY_FIELDNAMES,
    CVRP_SUMMARY_FIELDNAMES,
    read_csv_rows,
    summarize_ackley_rows,
    summarize_cvrp_rows,
    write_summary_csv,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TINY_INSTANCE = REPO_ROOT / "examples" / "tiny_cvrp.vrp"
BKS_PATH = REPO_ROOT / "data" / "cvrp_bks.csv"


def cvrp_row(instance="tiny", algorithm="sa", seed=42, best_cost="100.0",
             gap="", feasible=True, elapsed="1.0", cpu="1.0"):
    return {"instance": instance, "algorithm": algorithm, "seed": seed,
            "best_cost": best_cost, "gap_percent": gap, "feasible": feasible,
            "elapsed_time": elapsed, "cpu_time": cpu}


# ---------- summary functions ----------

def test_summarize_cvrp_rows_groups_and_stats():
    rows = [
        cvrp_row(seed=1, best_cost="100.0"),
        cvrp_row(seed=2, best_cost="80.0"),
        cvrp_row(algorithm="tabu", seed=1, best_cost="90.0"),
    ]
    summary = summarize_cvrp_rows(rows)
    assert len(summary) == 2  # (tiny, sa) and (tiny, tabu)
    sa = next(s for s in summary if s["algorithm"] == "sa")
    assert sa["runs"] == 2
    assert sa["feasible_runs"] == 2
    assert sa["best_cost"] == "80.0000"
    assert sa["mean_cost"] == "90.0000"
    assert float(sa["std_cost"]) == pytest.approx(14.1421, abs=1e-3)
    assert sa["best_seed"] == 2


def test_summarize_cvrp_rows_handles_missing_gap():
    summary = summarize_cvrp_rows([cvrp_row(gap="")])
    assert summary[0]["best_gap_percent"] == ""
    assert summary[0]["mean_gap_percent"] == ""

    with_gap = summarize_cvrp_rows([cvrp_row(gap="5.0"), cvrp_row(seed=2, gap="3.0")])
    assert with_gap[0]["best_gap_percent"] == "3.0000"


def test_summarize_ackley_rows():
    rows = [
        {"algorithm": "sa", "seed": 1, "best_value": "10.0",
         "distance_from_origin": "5.0", "elapsed_time": "1.0", "cpu_time": "1.0"},
        {"algorithm": "sa", "seed": 2, "best_value": "6.0",
         "distance_from_origin": "3.0", "elapsed_time": "1.0", "cpu_time": "1.0"},
        {"algorithm": "tabu", "seed": 1, "best_value": "8.0",
         "distance_from_origin": "4.0", "elapsed_time": "1.0", "cpu_time": "1.0"},
    ]
    summary = summarize_ackley_rows(rows)
    assert len(summary) == 2
    sa = next(s for s in summary if s["algorithm"] == "sa")
    assert sa["runs"] == 2
    assert sa["best_value"] == "6.000000"
    assert sa["mean_best_value"] == "8.000000"
    assert float(sa["std_best_value"]) == pytest.approx(2.8284, abs=1e-3)
    assert sa["best_seed"] == 2


def test_read_and_write_csv(tmp_path):
    rows = summarize_cvrp_rows([cvrp_row()])
    path = tmp_path / "sub" / "summary.csv"
    write_summary_csv(path, rows, CVRP_SUMMARY_FIELDNAMES)
    read_back = read_csv_rows(path)
    assert len(read_back) == 1
    assert read_back[0]["algorithm"] == "sa"


# ---------- CLI helpers ----------

def run_script(script_name, *extra_args):
    script = REPO_ROOT / "scripts" / script_name
    return subprocess.run(
        [sys.executable, str(script), *extra_args],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )


# ---------- BKS support in run_experiments.py ----------

def test_run_experiments_with_bks(tmp_path):
    output = tmp_path / "with_bks.csv"
    proc = run_script("run_experiments.py", "--part", "cvrp",
                      "--instances", str(TINY_INSTANCE), "--algorithms", "sa",
                      "--seeds", "42", "--budget", "5", "--timeout", "5",
                      "--bks", str(BKS_PATH), "--output", str(output))
    assert proc.returncode == 0
    rows = read_csv_rows(output)
    assert len(rows) == 1
    # tiny-cvrp has no BKS entry, so the gap stays empty
    assert rows[0]["gap_percent"] == ""


def test_run_experiments_unknown_algorithm_still_fails(tmp_path):
    proc = run_script("run_experiments.py", "--part", "cvrp",
                      "--instances", str(TINY_INSTANCE),
                      "--algorithms", "unknown_name", "--seeds", "42",
                      "--output", str(tmp_path / "bad.csv"))
    assert proc.returncode == 1


# ---------- smoke suite ----------

def test_smoke_suite(tmp_path):
    output_dir = tmp_path / "smoke"
    proc = run_script("run_smoke_suite.py", "--output-dir", str(output_dir),
                      "--cvrp-algorithms", "sa", "tabu",
                      "--ackley-algorithms", "sa", "tabu",
                      "--seeds", "42", "--budget", "5", "--timeout", "5",
                      "--ackley-dimension", "4")
    assert proc.returncode == 0
    assert (output_dir / "raw" / "cvrp_results.csv").exists()
    assert (output_dir / "raw" / "ackley_results.csv").exists()
    assert (output_dir / "summary" / "cvrp_summary.csv").exists()
    assert (output_dir / "summary" / "ackley_summary.csv").exists()
    assert list((output_dir / "cvrp_solutions").glob("*.txt"))

    cvrp_summary = read_csv_rows(output_dir / "summary" / "cvrp_summary.csv")
    assert len(cvrp_summary) == 2  # sa and tabu on the tiny instance
    assert all(row["feasible_runs"] == "1" for row in cvrp_summary)


def test_summarize_cli(tmp_path):
    output_dir = tmp_path / "smoke"
    run_script("run_smoke_suite.py", "--output-dir", str(output_dir),
               "--cvrp-algorithms", "sa", "--ackley-algorithms", "sa",
               "--seeds", "42", "--budget", "5", "--timeout", "5",
               "--ackley-dimension", "4")

    cvrp_out = tmp_path / "cvrp_summary2.csv"
    proc = run_script("summarize_experiments.py", "--part", "cvrp",
                      "--input", str(output_dir / "raw" / "cvrp_results.csv"),
                      "--output", str(cvrp_out))
    assert proc.returncode == 0
    assert read_csv_rows(cvrp_out)

    ackley_out = tmp_path / "ackley_summary2.csv"
    proc = run_script("summarize_experiments.py", "--part", "ackley",
                      "--input", str(output_dir / "raw" / "ackley_results.csv"),
                      "--output", str(ackley_out))
    assert proc.returncode == 0
    rows = read_csv_rows(ackley_out)
    assert rows and set(ACKLEY_SUMMARY_FIELDNAMES) == set(rows[0].keys())
