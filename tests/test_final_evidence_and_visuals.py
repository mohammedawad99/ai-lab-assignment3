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
    "final_v3_summary.txt",
    "cvrp_seed_robustness_summary.csv",
    # explicit ILS evidence (Stage 13-A)
    "cvrp_ils_runs.csv",
    "cvrp_ils_summary.csv",
    "cvrp_ils_manifest.json",
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
    "cvrp_stage11_advanced_impact.png",
    "rushhour_manual_heuristic_ladder.png",
    "rushhour_gp_gep_vs_manual.png",
    "rushhour_gp_gep_seed_variance.png",
    "cvrp_ils_comparison.png",
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
    text = REPORT_MD.read_text(encoding="utf-8")
    for name in REQUIRED_REFERENCES:
        assert f"figures/{name}" in text, name


def test_report_still_honest():
    text = REPORT_MD.read_text(encoding="utf-8")
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


def test_stage11_report_claims_match_evidence():
    # the Stage 11 project-best gaps in the report must equal the evidence
    import csv
    text = REPORT_MD.read_text(encoding="utf-8")
    with open(EVIDENCE / "cvrp_algorithm_mean_gaps.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    instances = [k for k in rows[0]
                 if k not in ("algorithm", "mean_best_gap_percent")]
    for inst in instances:
        best = min(float(r[inst]) for r in rows)
        assert f"{best:.4f}" in text, f"{inst}: {best:.4f} not in report"
    # the disclosed ALNS regression on X must stay disclosed
    assert "25.88" in text
    # advanced pass is explained
    assert "2-opt*" in text and "Or-opt" in text


def test_bnb_report_claims_match_evidence():
    # the report's B&B/LDS numbers must be backed by the committed evidence
    import csv
    text = REPORT_MD.read_text(encoding="utf-8")
    with open(EVIDENCE / "cvrp_algorithm_mean_gaps.csv", newline="") as f:
        by_algo = {r["algorithm"]: r for r in csv.DictReader(f)}
    bnb, base = by_algo["cvrp_bnb_lds"], by_algo["baseline"]
    instances = [k for k in bnb if k not in ("algorithm", "mean_best_gap_percent")]
    beats = sum(1 for i in instances if float(bnb[i]) < float(base[i]) - 1e-9)
    mean = sum(float(bnb[i]) for i in instances) / len(instances)
    assert f"{mean:.2f}%" in text
    assert f"{beats}/6" in text
    # the small-instance claims must be true in the evidence, not just prose
    assert float(bnb["P-n16-k8"]) < 0.5
    assert float(bnb["E-n22-k4"]) < 0.5
