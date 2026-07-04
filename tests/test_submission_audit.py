"""Tests for the submission audit and the final report PDF."""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = REPO_ROOT / "report" / "assignment3_report.pdf"
RESULTS_PRESENT = (REPO_ROOT / "results" / "final_experiments" / "raw"
                   / "cvrp_all_instances.csv").exists()


def run_audit(*extra_args):
    script = REPO_ROOT / "scripts" / "audit_submission.py"
    return subprocess.run(
        [sys.executable, str(script), *extra_args],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )


@pytest.mark.skipif(not (PDF_PATH.exists() and RESULTS_PRESENT),
                    reason="final PDF or final results not present")
def test_full_audit_passes():
    proc = run_audit("--check-results", "--check-pdf")
    assert proc.returncode == 0, proc.stdout
    assert "audit: PASS" in proc.stdout


def test_audit_checks_report_placeholders():
    proc = run_audit()
    assert "report has no unresolved placeholders" in proc.stdout


@pytest.mark.skipif(not RESULTS_PRESENT, reason="final results not present")
def test_audit_checks_row_counts():
    proc = run_audit("--check-results")
    assert "cvrp_all_instances has 126 or 144 rows" in proc.stdout
    assert "ackley_d10 has 21 rows" in proc.stdout
    assert "gp_gep_comparison has 6 rows" in proc.stdout


def test_audit_does_not_require_tracked_official_data():
    # the audit only prints a note about official .vrp files
    proc = run_audit()
    assert "official .vrp files present" in proc.stdout
    assert "NOTE" in proc.stdout


@pytest.mark.skipif(not PDF_PATH.exists(), reason="PDF not exported")
def test_pdf_exists_and_is_not_tiny():
    # the visual export with embedded figures is ~450 KB; the old text-only
    # export was ~34 KB, so 100 KB separates the two
    assert PDF_PATH.stat().st_size > 100_000


def test_no_forbidden_root_files():
    assert not (REPO_ROOT / "report.md").exists()
    assert not (REPO_ROOT / "report.pdf").exists()
    assert not (REPO_ROOT / "docs").exists()
    assert not (REPO_ROOT / "AI_USAGE.md").exists()
    zips = [p for p in REPO_ROOT.rglob("*.zip")
            if ".venv" not in p.parts and ".git" not in p.parts]
    assert not zips
