"""Tests for the official CVRP benchmark setup layer."""

import subprocess
import sys
from pathlib import Path

import pytest

from src.experiments.official_benchmarks import (
    OFFICIAL_CVRP_INSTANCES,
    check_official_data,
    expected_instance_path,
    find_official_instance_files,
    load_bks_table,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
BKS_PATH = REPO_ROOT / "data" / "cvrp_bks.csv"

EXPECTED_BKS = {
    "P-n16-k8": 450.0,
    "E-n22-k4": 375.0,
    "A-n32-k5": 784.0,
    "A-n80-k10": 1763.0,
    "X-n101-k25": 27591.0,
    "M-n200-k17": 1275.0,
}


def test_official_instance_list():
    assert len(OFFICIAL_CVRP_INSTANCES) == 6
    assert OFFICIAL_CVRP_INSTANCES == list(EXPECTED_BKS)


def test_load_bks_table():
    table = load_bks_table(BKS_PATH)
    assert table == EXPECTED_BKS
    assert all(cost > 0 for cost in table.values())


def test_load_bks_table_rejects_unknown(tmp_path):
    bad = tmp_path / "bks.csv"
    bad.write_text("instance,bks_cost\nSome-other-instance,100\n")
    with pytest.raises(ValueError):
        load_bks_table(bad)


def test_expected_instance_path():
    path = expected_instance_path("A-n32-k5", "some/dir")
    assert path == Path("some/dir") / "A-n32-k5.vrp"


def test_check_non_strict_passes_without_files(tmp_path):
    empty_dir = tmp_path / "official"
    empty_dir.mkdir()
    report = check_official_data(data_dir=empty_dir, bks_path=BKS_PATH, strict=False)
    assert report["ok"] is True
    assert report["found_instances"] == []
    assert len(report["missing_instances"]) == 6
    assert report["unexpected_files"] == []
    assert report["bks"] == EXPECTED_BKS


def test_check_strict_fails_without_files(tmp_path):
    empty_dir = tmp_path / "official"
    empty_dir.mkdir()
    report = check_official_data(data_dir=empty_dir, bks_path=BKS_PATH, strict=True)
    assert report["ok"] is False
    assert len(report["missing_instances"]) == 6


def test_unexpected_file_fails(tmp_path):
    data_dir = tmp_path / "official"
    data_dir.mkdir()
    (data_dir / "unexpected.vrp").write_text("NAME : unexpected\n")
    report = check_official_data(data_dir=data_dir, bks_path=BKS_PATH, strict=False)
    assert report["ok"] is False
    assert report["unexpected_files"] == ["unexpected.vrp"]
    assert find_official_instance_files(data_dir) == {}


# ---------- CLI ----------

def run_cli(*extra_args):
    script = REPO_ROOT / "scripts" / "check_official_cvrp_data.py"
    return subprocess.run(
        [sys.executable, str(script), *extra_args],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )


def test_cli_non_strict(tmp_path):
    empty_dir = tmp_path / "official"
    empty_dir.mkdir()
    proc = run_cli("--data-dir", str(empty_dir))
    assert proc.returncode == 0
    assert "ok: yes" in proc.stdout


def test_cli_csv_output(tmp_path):
    empty_dir = tmp_path / "official"
    empty_dir.mkdir()
    csv_path = tmp_path / "check.csv"
    proc = run_cli("--data-dir", str(empty_dir), "--csv-output", str(csv_path))
    assert proc.returncode == 0
    lines = csv_path.read_text().splitlines()
    assert lines[0] == "instance,expected_path,exists,bks_cost"
    assert len(lines) == 7  # header + 6 instances


def test_cli_strict_missing_files(tmp_path):
    empty_dir = tmp_path / "official"
    empty_dir.mkdir()
    proc = run_cli("--data-dir", str(empty_dir), "--strict")
    assert proc.returncode == 1
    assert "ok: no" in proc.stdout
