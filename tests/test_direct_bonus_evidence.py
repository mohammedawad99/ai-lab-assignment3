"""Stage 12-C: the direct no-A* bonus evidence must back the report claims."""

import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = REPO_ROOT / "report" / "evidence"
REPORT = REPO_ROOT / "report" / "assignment3_report.md"

DIRECT_FILES = [
    "direct_gp_gep_runs.csv",
    "direct_gp_gep_summary.csv",
    "direct_manual_baselines.csv",
    "direct_manual_summary.csv",
    "direct_planner_manifest.json",
]


def read_rows(name):
    with open(EVIDENCE / name, newline="") as f:
        return list(csv.DictReader(f))


def test_direct_evidence_files_exist_and_parse():
    for name in DIRECT_FILES:
        path = EVIDENCE / name
        assert path.exists(), name
        assert path.stat().st_size > 50, name
    assert read_rows("direct_gp_gep_runs.csv")
    assert read_rows("direct_manual_baselines.csv")


def test_direct_manifest_is_the_real_benchmark():
    manifest = json.loads((EVIDENCE / "direct_planner_manifest.json").read_text())
    assert manifest["smoke"] is False
    assert manifest["seeds"] == [42, 43, 44]
    assert manifest["population"] == 30
    assert manifest["generations"] == 20
    assert "no A*" in manifest["note"]


def test_gp_direct_beats_best_manual_direct():
    runs = read_rows("direct_gp_gep_runs.csv")
    baselines = read_rows("direct_manual_baselines.csv")
    gp_best = max(int(r["eval_solved"]) for r in runs
                  if r["method"] == "gp_direct")
    manual_best = max(int(r["eval_solved"]) for r in baselines)
    assert gp_best >= manual_best  # the bonus claim in the report


def test_gep_direct_reported_honestly():
    runs = read_rows("direct_gp_gep_runs.csv")
    gep_best = max(int(r["eval_solved"]) for r in runs
                   if r["method"] == "gep_direct")
    text = REPORT.read_text()
    assert f"{gep_best}/14" in text
    # the honest detail: GEP converged to the manual blocker_depth feature
    assert "blocker_depth" in text


def test_astar_reference_remains_stronger():
    runs = read_rows("direct_gp_gep_runs.csv")
    hard = read_rows("rushhour_hard_gp_gep_summary.csv")
    direct_best = max(int(r["eval_solved"]) for r in runs)
    astar_best = max(int(r["solved"]) for r in hard
                     if not r["name"].startswith("manual_"))
    assert astar_best > direct_best  # no overclaiming in either direction
    assert f"{astar_best}/14" in REPORT.read_text()


def test_report_covers_direct_bonus():
    text = REPORT.read_text()
    assert "without A*" in text
    assert "rushhour_direct_solved_counts.png" in text
    assert "rushhour_direct_gp_gep_fitness.png" in text
    for value in ("11/14", "9/14", "13/14", "120736", "97758"):
        assert value in text, value
