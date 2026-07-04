"""Tests for the plot helpers and report asset generation."""

import csv
import subprocess
import sys
from pathlib import Path

from src.experiments.plot_utils import (
    save_bar_chart,
    save_grouped_bar_chart,
    save_line_chart,
    save_text_note,
)
from src.experiments.report_assets import (
    load_rows_if_exists,
    make_ackley_report_assets,
    make_cvrp_report_assets,
    make_gp_gep_report_assets,
    rows_to_markdown_table,
    write_markdown_table,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


# ---------- plot utilities ----------

def test_save_bar_chart(tmp_path):
    path = save_bar_chart(["a", "b"], [1.0, 2.0], "t", "y", tmp_path / "bar.png")
    assert path.exists() and path.suffix == ".png"


def test_save_grouped_bar_chart(tmp_path):
    path = save_grouped_bar_chart(["g1", "g2"], {"s1": [1, 2], "s2": [3, 4]},
                                  "t", "y", tmp_path / "grouped.png")
    assert path.exists()


def test_save_line_chart(tmp_path):
    path = save_line_chart([1, 2, 3], [4, 5, 6], "t", "x", "y", tmp_path / "line.png")
    assert path.exists()


def test_save_text_note(tmp_path):
    path = save_text_note("hello", tmp_path / "note.txt")
    assert path.read_text() == "hello\n"


# ---------- markdown helpers ----------

def test_markdown_table_empty():
    assert rows_to_markdown_table([], ["a", "b"]) == "No rows available."


def test_markdown_table_rows():
    text = rows_to_markdown_table([{"a": 1, "b": "x"}], ["a", "b"])
    assert "| a | b |" in text
    assert "| 1 | x |" in text


def test_write_markdown_table(tmp_path):
    path = write_markdown_table(tmp_path / "table.md", [{"a": 1}], ["a"], title="T")
    assert path.exists()
    assert path.read_text().startswith("# T")


def test_load_rows_if_exists_missing(tmp_path):
    assert load_rows_if_exists(tmp_path / "nope.csv") == []
    assert load_rows_if_exists(None) == []


# ---------- CVRP assets ----------

CVRP_SUMMARY_ROWS = [
    {"instance": "tiny", "algorithm": "sa", "runs": 1, "feasible_runs": 1,
     "best_cost": "80.6450", "mean_cost": "80.6450", "std_cost": "0.0",
     "best_gap_percent": "", "mean_gap_percent": "",
     "mean_elapsed_time": "0.01", "mean_cpu_time": "0.01", "best_seed": 42},
]


def test_cvrp_assets_missing_inputs(tmp_path):
    created = make_cvrp_report_assets(tmp_path / "no_raw.csv",
                                      tmp_path / "no_summary.csv",
                                      tmp_path / "assets")
    assert len(created) == 1
    assert created[0].name == "cvrp_data_missing.txt"


def test_cvrp_assets_from_summary(tmp_path):
    summary = write_csv(tmp_path / "summary.csv", list(CVRP_SUMMARY_ROWS[0]),
                        CVRP_SUMMARY_ROWS)
    created = make_cvrp_report_assets(None, summary, tmp_path / "assets")
    names = [p.name for p in created]
    assert "cvrp_summary_table.md" in names
    assert any(name.endswith(".png") for name in names)


def test_cvrp_assets_from_raw_only(tmp_path):
    raw_rows = [{"instance": "tiny", "algorithm": "sa", "seed": 42,
                 "best_cost": "80.6450", "gap_percent": "", "feasible": True,
                 "elapsed_time": "0.01", "cpu_time": "0.01"}]
    raw = write_csv(tmp_path / "raw.csv", list(raw_rows[0]), raw_rows)
    created = make_cvrp_report_assets(raw, None, tmp_path / "assets")
    names = [p.name for p in created]
    assert "cvrp_summary_generated.csv" in names
    assert "cvrp_summary_table.md" in names


# ---------- Ackley assets ----------

ACKLEY_SUMMARY_ROWS = [
    {"algorithm": "sa", "runs": 1, "best_value": "3.5", "mean_best_value": "3.5",
     "std_best_value": "0.0", "mean_distance_from_origin": "2.0",
     "mean_elapsed_time": "0.01", "mean_cpu_time": "0.01", "best_seed": 42},
]


def test_ackley_assets_missing_inputs(tmp_path):
    created = make_ackley_report_assets(tmp_path / "no_raw.csv",
                                        tmp_path / "no_summary.csv",
                                        tmp_path / "assets")
    assert len(created) == 1
    assert created[0].name == "ackley_data_missing.txt"


def test_ackley_assets_from_summary(tmp_path):
    summary = write_csv(tmp_path / "summary.csv", list(ACKLEY_SUMMARY_ROWS[0]),
                        ACKLEY_SUMMARY_ROWS)
    created = make_ackley_report_assets(None, summary, tmp_path / "assets")
    names = [p.name for p in created]
    assert "ackley_summary_table.md" in names
    assert any(name.endswith(".png") for name in names)


def test_ackley_assets_from_raw_only(tmp_path):
    raw_rows = [{"algorithm": "sa", "seed": 42, "best_value": "3.5",
                 "distance_from_origin": "2.0", "elapsed_time": "0.01",
                 "cpu_time": "0.01"}]
    raw = write_csv(tmp_path / "raw.csv", list(raw_rows[0]), raw_rows)
    created = make_ackley_report_assets(raw, None, tmp_path / "assets")
    names = [p.name for p in created]
    assert "ackley_summary_generated.csv" in names
    assert "ackley_summary_table.md" in names


# ---------- GP/GEP assets ----------

def test_gp_gep_assets_missing_input(tmp_path):
    created = make_gp_gep_report_assets(tmp_path / "no.csv", None, tmp_path / "assets")
    assert len(created) == 1
    assert created[0].name == "gp_gep_data_missing.txt"


def test_gp_gep_assets_from_csv(tmp_path):
    rows = [{"algorithm": "gp_rushhour", "seed": 42, "train_fitness": "39956.0",
             "eval_fitness": "39956.0", "eval_solved_count": 4,
             "eval_puzzle_count": 4, "eval_total_expanded_nodes": 4,
             "eval_total_cost": 4, "best_expression": "blocking"}]
    comparison = write_csv(tmp_path / "comparison.csv", list(rows[0]), rows)
    created = make_gp_gep_report_assets(comparison, None, tmp_path / "assets")
    names = [p.name for p in created]
    assert "gp_gep_comparison_table.md" in names
    assert any(name.endswith(".png") for name in names)
    assert "gp_gep_summary_note.txt" in names  # missing-summary note


# ---------- CLI ----------

def run_cli(*extra_args):
    script = REPO_ROOT / "scripts" / "generate_report_assets.py"
    return subprocess.run(
        [sys.executable, str(script), *extra_args],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )


def test_cli_with_missing_inputs(tmp_path):
    proc = run_cli("--cvrp-raw", str(tmp_path / "no.csv"),
                   "--output-dir", str(tmp_path / "assets"))
    assert proc.returncode == 0
    assert (tmp_path / "assets" / "cvrp_data_missing.txt").exists()


def test_cli_with_summary_inputs(tmp_path):
    summary = write_csv(tmp_path / "cvrp_summary.csv", list(CVRP_SUMMARY_ROWS[0]),
                        CVRP_SUMMARY_ROWS)
    proc = run_cli("--cvrp-summary", str(summary),
                   "--output-dir", str(tmp_path / "assets"))
    assert proc.returncode == 0
    assert "assets created:" in proc.stdout
    assert (tmp_path / "assets" / "cvrp_summary_table.md").exists()
