"""Tests for the final execution runner (no long experiments are run here)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.experiments.final_execution import (
    aggregate_final_results,
    combine_csv_files,
    run_final_experiment_suite,
    write_execution_manifest,
)
from src.experiments.final_plan import load_final_plan
from src.experiments.official_benchmarks import OFFICIAL_CVRP_INSTANCES

REPO_ROOT = Path(__file__).resolve().parents[1]

OFFICIAL_DATA_PRESENT = all(
    (REPO_ROOT / "data" / "official_cvrp" / f"{name}.vrp").exists()
    for name in OFFICIAL_CVRP_INSTANCES
)


def write_small_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "a,b\n"
    path.write_text(header + "".join(f"{r[0]},{r[1]}\n" for r in rows))
    return path


def make_tmp_plan(tmp_path):
    """Copy of the real plan with the output dir moved into tmp_path."""
    plan = load_final_plan(REPO_ROOT / "configs" / "final_experiment_plan.json")
    plan["output_dir"] = str(tmp_path / "final")
    plan["assets"] = {"output_dir": str(tmp_path / "final" / "report_assets")}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    return plan, plan_path


# ---------- csv combination and manifest ----------

def test_combine_csv_files(tmp_path):
    one = write_small_csv(tmp_path / "one.csv", [(1, 2)])
    two = write_small_csv(tmp_path / "two.csv", [(3, 4), (5, 6)])
    info = combine_csv_files([one, two], tmp_path / "combined.csv")
    assert info["input_count"] == 2
    assert info["missing_count"] == 0
    assert info["row_count"] == 3
    lines = (tmp_path / "combined.csv").read_text(encoding="utf-8").splitlines()
    assert lines == ["a,b", "1,2", "3,4", "5,6"]  # exactly one header


def test_combine_csv_files_reports_missing(tmp_path):
    one = write_small_csv(tmp_path / "one.csv", [(1, 2)])
    info = combine_csv_files([one, tmp_path / "gone.csv"], tmp_path / "combined.csv")
    assert info["input_count"] == 1
    assert info["missing_count"] == 1
    assert str(tmp_path / "gone.csv") in info["missing_paths"]
    assert info["row_count"] == 1


def test_write_execution_manifest(tmp_path):
    path = write_execution_manifest(tmp_path / "sub" / "manifest.json", {"x": 1})
    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8")) == {"x": 1}


# ---------- aggregation with missing files ----------

def test_aggregate_handles_missing_raw_files(tmp_path):
    plan, _ = make_tmp_plan(tmp_path)
    info = aggregate_final_results(plan, tmp_path / "final")
    # nothing was run, so everything is missing but nothing crashes
    assert info["cvrp_combined"]["missing_count"] == 6
    assert info["cvrp_combined"]["row_count"] == 0
    assert len(info["missing"]) >= 6  # per-instance raws + ackley raw


# ---------- suite with all parts skipped ----------

@pytest.mark.skipif(not OFFICIAL_DATA_PRESENT,
                    reason="official .vrp files not placed under data/official_cvrp/")
def test_suite_with_all_parts_skipped(tmp_path):
    _, plan_path = make_tmp_plan(tmp_path)
    manifest = run_final_experiment_suite(plan_path=plan_path, resume=True,
                                          run_cvrp=False, run_ackley=False,
                                          run_rushhour=False)
    for key in ("cvrp_run_info", "ackley_run_info", "rushhour_run_info",
                "aggregation_info", "asset_paths"):
        assert key in manifest
    assert manifest["cvrp_run_info"] == []
    assert Path(manifest["manifest_path"]).exists()
    # missing-data note assets were still created
    assert manifest["asset_paths"]


@pytest.mark.skipif(not OFFICIAL_DATA_PRESENT,
                    reason="official .vrp files not placed under data/official_cvrp/")
def test_cli_with_all_parts_skipped(tmp_path):
    _, plan_path = make_tmp_plan(tmp_path)
    script = REPO_ROOT / "scripts" / "run_final_experiments.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--plan", str(plan_path),
         "--skip-cvrp", "--skip-ackley", "--skip-rushhour"],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert proc.returncode == 0
    assert "manifest:" in proc.stdout
