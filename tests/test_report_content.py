"""Checks for the filled final report."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = REPO_ROOT / "report" / "assignment3_report.md"

OFFICIAL_INSTANCES = ["P-n16-k8", "E-n22-k4", "A-n32-k5", "A-n80-k10",
                      "X-n101-k25", "M-n200-k17"]
BKS_VALUES = ["450", "375", "784", "1763", "27591", "1275"]


def report_text():
    return REPORT_PATH.read_text(encoding="utf-8")


def test_report_exists_with_headings():
    text = report_text()
    for heading in [
        "# Assignment 3 Report",
        "## 1. Introduction",
        "## 4. Part A — CVRP",
        "## 6. CVRP Algorithms",
        "## 8. Part B — Rush Hour with GP and GEP",
        "## 10. Analysis and Discussion",
        "## 12. Use of AI Tools",
        "## 13. Reproducibility",
        "## 14. Conclusion",
    ]:
        assert heading in text, heading


def test_report_mentions_instances_and_bks():
    text = report_text()
    for name in OFFICIAL_INSTANCES:
        assert name in text
    for value in BKS_VALUES:
        assert value in text


def test_report_core_topics():
    text = report_text()
    assert "Ackley" in text
    assert "d = 10" in text
    assert "GP" in text and "GEP" in text
    assert "A*" in text
    assert "AI tools" in text


def test_report_has_no_unresolved_placeholders():
    assert report_text().count("[fill after final run]") == 0


def test_report_makes_no_overclaims():
    text = report_text().lower()
    for phrase in ["prove optimal", "globally best", "guaranteed best"]:
        assert phrase not in text


def test_report_asset_paths_are_under_results():
    text = report_text()
    if "report_assets" in text:
        assert "results/final_experiments/report_assets" in text


def test_no_forbidden_report_files():
    assert not (REPO_ROOT / "report.pdf").exists()
    assert not (REPO_ROOT / "report.md").exists()  # root-level
