"""Checks for the visual report: figures, references, and the exported PDF."""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_MD = REPO_ROOT / "report" / "assignment3_report.md"
REPORT_PDF = REPO_ROOT / "report" / "assignment3_report.pdf"
FIGURES = REPO_ROOT / "report" / "figures"

REQUIRED_FIGURES = [
    "cvrp_best_gap_by_instance.png",
    "cvrp_mean_gap_by_algorithm.png",
    "cvrp_runtime_by_instance.png",
    "ackley_best_value_by_algorithm.png",
    "ackley_runtime_by_algorithm.png",
    "gp_gep_eval_fitness.png",
    "gp_gep_expanded_nodes.png",
    "code_baseline_repair.png",
    "code_gep_decoder.png",
    "code_final_runner.png",
    "terminal_audit_pass.png",
    "terminal_final_rows.png",
]


def test_scripts_exist():
    assert (REPO_ROOT / "scripts" / "generate_report_figures.py").exists()
    assert (REPO_ROOT / "scripts" / "export_report_pdf.py").exists()


def test_all_required_figures_exist():
    assert FIGURES.is_dir()
    for name in REQUIRED_FIGURES:
        path = FIGURES / name
        assert path.exists(), name
        assert path.stat().st_size > 5_000, name  # a real image, not a stub


def test_report_references_figures():
    text = REPORT_MD.read_text()
    references = [name for name in REQUIRED_FIGURES
                  if f"figures/{name}" in text]
    assert len(references) >= 8, references


def test_report_content_still_honest():
    text = REPORT_MD.read_text()
    assert "[fill after final run]" not in text
    assert "X-n101-k25" in text
    assert "25.45" in text  # the weak gap stays visible


@pytest.mark.skipif(not REPORT_PDF.exists(), reason="PDF not exported")
def test_pdf_is_visual():
    data = REPORT_PDF.read_bytes()
    assert len(data) > 100_000  # text-only export was ~34 KB
    pages = data.count(b"/Type /Page") - data.count(b"/Type /Pages")
    assert pages >= 10
    assert data.count(b"/Subtype /Image") >= 8  # embedded figures


def test_no_forbidden_files():
    assert not (REPO_ROOT / "report.md").exists()
    assert not (REPO_ROOT / "report.pdf").exists()
    assert not (REPO_ROOT / "docs").exists()
    assert not (REPO_ROOT / "AI_USAGE.md").exists()
    zips = [p for p in REPO_ROOT.rglob("*.zip")
            if ".venv" not in p.parts and ".git" not in p.parts]
    assert not zips
