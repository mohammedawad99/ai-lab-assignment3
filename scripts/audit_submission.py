"""Local audit for the final submission state.

Checks that the tracked project files, the filled report, and (optionally)
the final result files and the PDF are all in the expected state. Prints one
PASS/FAIL line per check and exits 0 only if everything required passed.

Usage:
    python scripts/audit_submission.py [--strict] [--check-results] [--check-pdf]
"""

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "README.md",
    "requirements.txt",
    "report/assignment3_report.md",
    "configs/final_experiment_plan.json",
    "data/cvrp_bks.csv",
    # small evidence snapshots cited by the report
    "report/evidence/cvrp_all_summary.csv",
    "report/evidence/cvrp_all_policy_effective_summary.csv",
    "report/evidence/ackley_d10_summary.csv",
    "report/evidence/gp_gep_comparison_runs.csv",
    "report/evidence/rushhour_hard_manual_summary.csv",
    "report/evidence/rushhour_hard_gp_gep_summary.csv",
    "report/evidence/final_execution_manifest.json",
    "report/evidence/final_v2_summary.txt",
    "report/evidence/cvrp_seed_robustness_summary.csv",
]

RESULT_FILES = [
    "results/final_experiments/raw/cvrp_all_instances.csv",
    "results/final_experiments/summary/cvrp_all_summary.csv",
    "results/final_experiments/raw/ackley_d10.csv",
    "results/final_experiments/summary/ackley_d10_summary.csv",
    "results/final_experiments/raw/gp_gep_comparison_runs.csv",
    "results/final_experiments/raw/gp_gep_comparison_summary.txt",
    "results/final_experiments/final_execution_manifest.json",
]

FORBIDDEN = ["docs", "AI_USAGE.md", "report.md", "report.pdf"]

OVERCLAIM_PHRASES = ["prove optimal", "globally best", "guaranteed best"]

# the removed benchmark instance name, assembled so the audit itself
# does not contain the literal string it scans for
REMOVED_INSTANCE = "-".join(["B", "n45", "k9"])

SKIP_DIRS = {".git", ".venv", "results", "__pycache__", ".pytest_cache"}
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".csv", ".vrp", ".cfg", ".toml", ".gitignore"}


def count_data_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description="Audit the submission state.")
    parser.add_argument("--strict", action="store_true",
                        help="treat optional notes as failures too")
    parser.add_argument("--check-results", action="store_true")
    parser.add_argument("--check-pdf", action="store_true")
    args = parser.parse_args()

    failures = []

    def check(name, ok, detail=""):
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f" ({detail})" if detail else ""))
        if not ok:
            failures.append(name)

    # required tracked files
    required = list(REQUIRED_FILES)
    if args.check_pdf:
        required.append("report/assignment3_report.pdf")
    for rel in required:
        check(f"file exists: {rel}", (REPO_ROOT / rel).exists())

    # report content
    report_path = REPO_ROOT / "report" / "assignment3_report.md"
    if report_path.exists():
        text = report_path.read_text()
        check("report has no unresolved placeholders",
              "[fill after final run]" not in text)
        lowered = text.lower()
        check("report has no overclaim phrases",
              all(p not in lowered for p in OVERCLAIM_PHRASES))
        check("report has an AI tools section", "AI tools" in text)
        check("report covers the tuned rerun", "tuned" in text)
        check("report covers the hard Rush Hour benchmark",
              "hard Rush Hour" in text and "blocker_depth" in text)
        check("report shows before/after tuning figure",
              "cvrp_before_after_tuning.png" in text)
        check("report covers seed robustness",
              "cvrp_seed_gap_boxplots.png" in text)

    # result files and row counts
    if args.check_results:
        for rel in RESULT_FILES:
            check(f"result exists: {rel}", (REPO_ROOT / rel).exists())
        cvrp_path = REPO_ROOT / RESULT_FILES[0]
        if cvrp_path.exists():
            rows = count_data_rows(cvrp_path)
            # 144 = 8 algorithm rows (incl. both ALNS variants) x 6 x 3
            check("cvrp_all_instances has 144 rows",
                  len(rows) == 144, f"{len(rows)} rows")
            check("all cvrp rows feasible",
                  all(str(r.get("feasible")).lower() in ("true", "yes", "1")
                      for r in rows))
            check("no cvrp row has errors", all(not r.get("errors") for r in rows))
        ackley_path = REPO_ROOT / RESULT_FILES[2]
        if ackley_path.exists():
            rows = count_data_rows(ackley_path)
            check("ackley_d10 has 21 rows", len(rows) == 21, f"{len(rows)} rows")
        gp_path = REPO_ROOT / RESULT_FILES[4]
        if gp_path.exists():
            rows = count_data_rows(gp_path)
            check("gp_gep_comparison has 6 rows", len(rows) == 6, f"{len(rows)} rows")

    # PDF sanity
    if args.check_pdf:
        pdf_path = REPO_ROOT / "report" / "assignment3_report.pdf"
        if pdf_path.exists():
            check("pdf size > 10 KB", pdf_path.stat().st_size > 10_000,
                  f"{pdf_path.stat().st_size} bytes")

    # forbidden root-level files/dirs
    for name in FORBIDDEN:
        check(f"forbidden absent: {name}", not (REPO_ROOT / name).exists())
    zips = [p for p in REPO_ROOT.rglob("*.zip")
            if ".venv" not in p.parts and ".git" not in p.parts]
    check("no ZIP files", not zips, ", ".join(str(p) for p in zips))

    # removed instance scan over repository text files
    hits = []
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        try:
            if REMOVED_INSTANCE in path.read_text(errors="ignore"):
                hits.append(str(path.relative_to(REPO_ROOT)))
        except OSError:
            continue
    check("removed instance name absent", not hits, ", ".join(hits))

    # official data note (informational, never required to be tracked)
    official = list((REPO_ROOT / "data" / "official_cvrp").glob("*.vrp"))
    print(f"NOTE  official .vrp files present: {len(official)} "
          "(user-provided, not required to be tracked)")

    print(f"\naudit: {'PASS' if not failures else 'FAIL'} "
          f"({len(failures)} failed check(s))")
    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    main()
