"""Checks for the Stage 9-F evidence snapshots and route/convergence visuals."""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_MD = REPO_ROOT / "report" / "assignment3_report.md"
REPORT_PDF = REPO_ROOT / "report" / "assignment3_report.pdf"
FIGURES = REPO_ROOT / "report" / "figures"
EVIDENCE = REPO_ROOT / "report" / "evidence"

EVIDENCE_FILES = [
    "cvrp_all_summary.csv",
    "cvrp_all_policy_effective_summary.csv",
    "cvrp_algorithm_mean_gaps.csv",
    "ackley_d10_summary.csv",
    "gp_gep_comparison_runs.csv",
    "rushhour_hard_manual_summary.csv",
    "rushhour_hard_gp_gep_summary.csv",
    "rushhour_hard_config.json",
    "rushhour_hard_manifest.json",
    "final_execution_manifest.json",
    "final_v2_summary.txt",
]

VISUAL_FIGURES = [
    "cvrp_route_P-n16-k8.png",
    "cvrp_route_A-n80-k10.png",
    "cvrp_route_X-n101-k25.png",
    "convergence_P-n16-k8.png",
    "convergence_A-n80-k10.png",
    "convergence_X-n101-k25.png",
    "alns_operator_weights_A-n80-k10.png",
]

REQUIRED_REFERENCES = [
    "cvrp_route_P-n16-k8.png",
    "cvrp_route_A-n80-k10.png",
    "cvrp_route_X-n101-k25.png",
    "convergence_P-n16-k8.png",
    "convergence_A-n80-k10.png",
    "cvrp_before_after_tuning.png",
    "rushhour_manual_heuristic_ladder.png",
    "rushhour_gp_gep_vs_manual.png",
    "rushhour_gp_gep_seed_variance.png",
]


def test_evidence_folder():
    assert EVIDENCE.is_dir()
    for name in EVIDENCE_FILES:
        path = EVIDENCE / name
        assert path.exists(), name
        assert path.stat().st_size > 100, name


def test_visual_scripts_exist():
    assert (REPO_ROOT / "scripts" / "generate_route_visualizations.py").exists()
    assert (REPO_ROOT / "scripts" / "generate_convergence_figures.py").exists()


def test_visual_figures_exist():
    for name in VISUAL_FIGURES:
        path = FIGURES / name
        assert path.exists(), name
        assert path.stat().st_size > 5_000, name


def test_report_references_new_figures():
    text = REPORT_MD.read_text()
    for name in REQUIRED_REFERENCES:
        assert f"figures/{name}" in text, name


def test_report_still_honest():
    text = REPORT_MD.read_text()
    assert "[fill after final run]" not in text
    assert "X-n101-k25" in text
    assert "25.45" in text


@pytest.mark.skipif(not REPORT_PDF.exists(), reason="PDF not exported")
def test_pdf_grew_with_visuals():
    assert REPORT_PDF.stat().st_size > 500_000


def test_no_full_results_or_official_data_committed():
    import subprocess
    tracked = subprocess.run(["git", "ls-files"], capture_output=True,
                             text=True, cwd=REPO_ROOT).stdout.splitlines()
    assert not any(p.startswith("results/") and p != "results/.gitkeep"
                   for p in tracked)
    assert not any(p.endswith(".vrp") and p.startswith("data/official_cvrp/")
                   for p in tracked)
