"""Stage 11-F: the report's CVRP claims must be backed by the committed
evidence CSVs, enforced by scripts/verify_report_matches_csv.py."""

import csv
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "verify_report_matches_csv.py"
EVIDENCE = REPO_ROOT / "report" / "evidence"
REPORT = REPO_ROOT / "report" / "assignment3_report.md"
INSTANCES = ["P-n16-k8", "E-n22-k4", "A-n32-k5", "A-n80-k10",
             "X-n101-k25", "M-n200-k17"]


def read_rows(name):
    with open(EVIDENCE / name, newline="") as f:
        return list(csv.DictReader(f))


def test_verifier_exists():
    assert SCRIPT.exists()


def test_verifier_passes_on_current_repo():
    result = subprocess.run([sys.executable, str(SCRIPT)],
                            capture_output=True, text=True, cwd=REPO_ROOT)
    assert result.returncode == 0, result.stdout + result.stderr


def test_headline_best_gaps_supported_by_summary_csv():
    text = REPORT.read_text(encoding="utf-8")
    rows = read_rows("cvrp_all_policy_effective_summary.csv")
    for inst in INSTANCES:
        best = min(float(r["best_gap_percent"]) for r in rows
                   if r["instance"] == inst)
        assert f"{best:.4f}" in text or f"{best:.2f}" in text, \
            f"{inst}: {best:.4f} not stated in report"


def test_mean_gaps_supported_by_mean_gaps_csv():
    text = REPORT.read_text(encoding="utf-8")
    for row in read_rows("cvrp_algorithm_mean_gaps.csv"):
        mean = float(row["mean_best_gap_percent"])
        assert f"{mean:.3f}" in text or f"{mean:.2f}" in text, \
            f"{row['algorithm']}: {mean:.3f} not stated in report"


def test_seed_source_statement_not_contradictory():
    text = REPORT.read_text(encoding="utf-8")
    # canonical 3-seed statement present
    assert "seeds 42, 43 and 44" in text
    assert "canonical source of every headline table" in text
    # canonical evidence really is 3-seed
    seeds = {r["best_seed"] for r in read_rows("cvrp_all_summary.csv")
             if r.get("best_seed")}
    assert seeds <= {"42", "43", "44"}, seeds
    # any 8-seed mention stays in the labeled robustness context
    if "42–49" in text or "42-49" in text:
        assert "robustness" in text and "not the source of any" in text


def test_bnb_consistency_still_enforced():
    text = REPORT.read_text(encoding="utf-8")
    by_algo = {r["algorithm"]: r for r in read_rows("cvrp_algorithm_mean_gaps.csv")}
    bnb, base = by_algo["cvrp_bnb_lds"], by_algo["baseline"]
    mean = sum(float(bnb[i]) for i in INSTANCES) / len(INSTANCES)
    beats = sum(1 for i in INSTANCES if float(bnb[i]) < float(base[i]) - 1e-9)
    assert f"{mean:.2f}%" in text
    assert f"{beats}/6" in text
