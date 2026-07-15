"""Local audit for the final submission state.

Checks that the tracked project files, the filled report, and (optionally)
the final result files and the PDF are all in the expected state. Prints one
PASS/FAIL line per check and exits 0 only if everything required passed.

Two modes:
- development audit (default): run in the full working repository;
  --check-results additionally verifies the generated files under
  results/final_experiments/.
- --submission-package: validates a clean clone / extracted source package,
  where results/ is intentionally absent. The canonical evidence under
  report/evidence/ and the PDF are required instead; --check-results is
  ignored in this mode.

Usage:
    python scripts/audit_submission.py [--strict] [--check-results] [--check-pdf]
    python scripts/audit_submission.py --submission-package [--check-pdf]
"""

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "README.md",
    "requirements.txt",
    # executable-runner compliance (Stage 12-A)
    "a3.py",
    "run_a3.sh",
    "run_a3.bat",
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
    "report/evidence/final_v3_summary.txt",
    "report/evidence/cvrp_seed_robustness_summary.csv",
    # bonus no-A* direct planner evidence (Stage 12-C)
    "report/evidence/direct_gp_gep_runs.csv",
    "report/evidence/direct_manual_baselines.csv",
    "report/evidence/direct_planner_manifest.json",
    # explicit ILS evidence (Stage 13-A)
    "report/evidence/cvrp_ils_runs.csv",
    "report/evidence/cvrp_ils_summary.csv",
    "report/evidence/cvrp_ils_manifest.json",
]

# Stage 11-C project-best gaps that the report must state (they must also
# match the evidence CSVs, checked below against cvrp_algorithm_mean_gaps)
STAGE11_BEST_GAPS = ["23.0063", "2.9466", "5.5139"]

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
    parser.add_argument("--submission-package", action="store_true",
                        help="validate a clean source package: report/evidence "
                             "and the PDF are required, results/ is not")
    args = parser.parse_args()

    if args.submission_package:
        # a clean package has no generated results/ tree by design
        if args.check_results:
            print("NOTE  --check-results ignored in --submission-package mode "
                  "(results/ is intentionally not part of the package)")
        args.check_results = False
        args.check_pdf = True

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
        text = report_path.read_text(encoding="utf-8")
        check("report has no unresolved placeholders",
              "[fill after final run]" not in text)
        lowered = text.lower()
        check("report has no overclaim phrases",
              all(p not in lowered for p in OVERCLAIM_PHRASES))
        check("report has an AI tools section", "AI tools" in text)
        # Stage 13-A: explicit ILS coverage and formal report elements
        check("report has an explicit ILS section",
              "Iterated Local Search (ILS)" in text)
        check("report uses exploration/exploitation vocabulary",
              "exploration" in lowered and "exploitation" in lowered)
        check("report has a references section", "## 15. References" in text)
        check("report has a cover identification block",
              "Submitted by" in text)
        check("report covers the tuned rerun", "tuned" in text)
        check("report covers the hard Rush Hour benchmark",
              "hard Rush Hour" in text and "blocker_depth" in text)
        check("report shows before/after tuning figure",
              "cvrp_before_after_tuning.png" in text)
        check("report covers seed robustness",
              "cvrp_seed_gap_boxplots.png" in text)

        # Stage 12-C: the direct no-A* bonus must be reported honestly
        direct_manifest = REPO_ROOT / "report" / "evidence" / "direct_planner_manifest.json"
        if direct_manifest.exists():
            import json as _json
            direct = _json.loads(direct_manifest.read_text(encoding="utf-8"))
            check("direct bonus manifest is not smoke",
                  direct.get("smoke") is False)
            check("report covers the direct no-A* bonus",
                  "without A*" in text and "11/14" in text
                  and "9/14" in text and "13/14" in text)
            check("report shows the direct bonus figure",
                  "rushhour_direct_solved_counts.png" in text)

        # Stage 11: advanced local-search results must be in the report
        for value in STAGE11_BEST_GAPS:
            check(f"report has Stage 11 best gap {value}", value in text)
        check("report explains advanced local-search",
              "advanced" in text.lower() and "2-opt*" in text
              and "candidate list" in text.lower().replace("candidate-list",
                                                           "candidate list"))
        check("report discloses the ALNS regression on X",
              "25.01" in text and "25.88" in text)
        check("report shows the advanced-impact figure",
              "cvrp_stage11_advanced_impact.png" in text)

        # B&B/LDS report-vs-evidence consistency (added after Stage 10-H
        # found the report ahead of the committed evidence)
        mean_gaps_path = REPO_ROOT / "report" / "evidence" / "cvrp_algorithm_mean_gaps.csv"
        all_summary_path = REPO_ROOT / "report" / "evidence" / "cvrp_all_summary.csv"
        if mean_gaps_path.exists() and all_summary_path.exists():
            by_algo = {r["algorithm"]: r for r in count_data_rows(mean_gaps_path)}
            bnb = by_algo.get("cvrp_bnb_lds")
            base = by_algo.get("baseline")
            if bnb and base:
                instances = [k for k in bnb
                             if k not in ("algorithm", "mean_best_gap_percent")]
                beats = sum(1 for i in instances
                            if float(bnb[i]) < float(base[i]) - 1e-9)
                # recompute the mean from the per-instance values so the
                # 3-decimal stored field cannot cause a rounding mismatch
                mean = sum(float(bnb[i]) for i in instances) / len(instances)
                mean_text = f"{mean:.2f}%"
                check("report B&B mean gap matches evidence",
                      mean_text in text, mean_text)
                check("report B&B beats-baseline count matches evidence",
                      f"{beats}/6" in text, f"evidence says {beats}/6")
                summary_bnb = {r["instance"]: float(r["best_gap_percent"])
                               for r in count_data_rows(all_summary_path)
                               if r["algorithm"] == "cvrp_bnb_lds"}
                consistent = all(abs(summary_bnb[i] - float(bnb[i])) < 0.01
                                 for i in instances if i in summary_bnb)
                check("B&B evidence CSVs agree with each other", consistent)

                # the report's per-instance project-best gaps must equal the
                # minimum over algorithms in the committed evidence
                for inst in instances:
                    best = min(float(row[inst]) for row in by_algo.values())
                    check(f"report best gap for {inst} matches evidence",
                          f"{best:.4f}" in text, f"{best:.4f}%")

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

    # executable-runner compliance (Stage 12-A)
    readme = REPO_ROOT / "README.md"
    if readme.exists():
        readme_text = readme.read_text(encoding="utf-8")
        check("README documents the executable run commands",
              "Executable / run commands" in readme_text
              and "a3.py" in readme_text)
    import subprocess as sp
    try:
        git_list = sp.run(["git", "ls-files"], capture_output=True, text=True,
                          cwd=REPO_ROOT)
        tracked = git_list.stdout.splitlines() if git_list.returncode == 0 else []
    except OSError:
        tracked = []  # no git available (e.g. extracted package)
    if args.submission_package:
        binary_hits = [str(p.relative_to(REPO_ROOT)) for pattern in
                       ("dist", "build") for p in [REPO_ROOT / pattern]
                       if p.exists()]
        binary_hits += [str(p.relative_to(REPO_ROOT))
                        for p in REPO_ROOT.rglob("*.exe")
                        if ".venv" not in p.parts and ".git" not in p.parts]
        check("no dist/build binaries in package", not binary_hits,
              ", ".join(binary_hits))
    else:
        check("no dist/build binaries tracked",
              not any(p.startswith(("dist/", "build/")) or p.endswith(".exe")
                      or p.endswith(".spec") for p in tracked))

    # report-vs-evidence consistency gate (Stage 11-F)
    import subprocess
    gate = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "verify_report_matches_csv.py")],
        capture_output=True, text=True, cwd=REPO_ROOT)
    check("report matches evidence CSVs (verify_report_matches_csv)",
          gate.returncode == 0,
          gate.stdout.strip().splitlines()[-1] if gate.stdout else "no output")

    # forbidden files/dirs. In the development repository the rule is that
    # they are never tracked by git (untracked local copies, e.g. a docs/
    # folder with the course PDFs kept only on this machine, are a note, not
    # a failure — a git-based package will not contain them). In a clean
    # extracted package there is no git metadata, so the files must simply
    # not exist.
    for name in FORBIDDEN:
        if args.submission_package:
            if (REPO_ROOT / ".git").exists() and (REPO_ROOT / name).exists() \
                    and not [p for p in tracked
                             if p == name or p.startswith(name + "/")]:
                # package mode run inside the dev repo: an untracked local
                # copy will not be in a git-based package
                print(f"NOTE  {name} exists locally but is untracked "
                      "(a git-based package will not contain it)")
                continue
            check(f"forbidden absent: {name}", not (REPO_ROOT / name).exists())
            continue
        tracked_hits = [p for p in tracked
                        if p == name or p.startswith(name + "/")]
        check(f"forbidden not tracked: {name}", not tracked_hits,
              ", ".join(tracked_hits))
        if not tracked_hits and (REPO_ROOT / name).exists():
            print(f"NOTE  {name} exists locally but is untracked "
                  "(will not be part of a git-based package)")
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
