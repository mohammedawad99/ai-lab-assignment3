"""Tests for the official mini-run helper (no official files required)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.experiments.official_mini_run import (
    available_official_instances,
    choose_official_mini_instances,
    run_official_mini_experiment,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
BKS_PATH = REPO_ROOT / "data" / "cvrp_bks.csv"
CONFIG_PATH = REPO_ROOT / "configs" / "official_mini_experiment.json"


@pytest.fixture
def empty_data_dir(tmp_path):
    data_dir = tmp_path / "official"
    data_dir.mkdir()
    return data_dir


def test_available_instances_empty_dir(empty_data_dir):
    assert available_official_instances(empty_data_dir, BKS_PATH) == []


def test_choose_with_no_files(empty_data_dir):
    selected, missing = choose_official_mini_instances(None, empty_data_dir, BKS_PATH)
    assert selected == []
    assert "P-n16-k8" in missing


def test_choose_rejects_unknown_instance(empty_data_dir):
    with pytest.raises(ValueError):
        choose_official_mini_instances(["Some-other-instance"], empty_data_dir, BKS_PATH)


def test_mini_run_allow_missing(empty_data_dir, tmp_path):
    result = run_official_mini_experiment(
        data_dir=empty_data_dir, bks_path=BKS_PATH,
        output_dir=tmp_path / "mini", allow_missing_data=True,
    )
    assert result["ok"] is True
    assert result["skipped"] is True
    assert result["reason"] == "missing official data"
    assert result["selected_instances"] == []
    assert result["raw_rows"] == []
    assert result["raw_csv"] == ""


def test_mini_run_raises_without_allow_missing(empty_data_dir, tmp_path):
    with pytest.raises(FileNotFoundError):
        run_official_mini_experiment(
            data_dir=empty_data_dir, bks_path=BKS_PATH,
            output_dir=tmp_path / "mini", allow_missing_data=False,
        )


def test_config_json_loads():
    config = json.loads(CONFIG_PATH.read_text())
    assert config["default_instance"] == "P-n16-k8"
    assert "readiness run" in config["note"]
    assert len(config["algorithms"]) == 7  # baseline + the six required


# ---------- CLI ----------

def run_cli(*extra_args):
    script = REPO_ROOT / "scripts" / "run_official_mini_experiment.py"
    return subprocess.run(
        [sys.executable, str(script), *extra_args],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )


def test_cli_allow_missing(empty_data_dir, tmp_path):
    proc = run_cli("--data-dir", str(empty_data_dir),
                   "--output-dir", str(tmp_path / "mini"), "--allow-missing-data")
    assert proc.returncode == 0
    assert "skipped: True" in proc.stdout


def test_cli_missing_without_flag(empty_data_dir, tmp_path):
    proc = run_cli("--data-dir", str(empty_data_dir),
                   "--output-dir", str(tmp_path / "mini"))
    assert proc.returncode == 1
    assert "missing" in proc.stderr


def test_cli_unknown_algorithm(empty_data_dir, tmp_path):
    proc = run_cli("--data-dir", str(empty_data_dir),
                   "--algorithms", "unknown_name",
                   "--output-dir", str(tmp_path / "mini"), "--allow-missing-data")
    assert proc.returncode == 1
    assert "unknown algorithm" in proc.stderr
