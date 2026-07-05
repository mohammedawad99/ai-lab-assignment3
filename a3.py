"""Unified command-line entry point for Assignment 3.

Thin wrapper: every subcommand delegates to the existing scripts under
scripts/ (no algorithm logic lives here). Run from the repository root:

    python a3.py --help
    python a3.py sanity
    python a3.py audit
    python a3.py test
    python a3.py report-info
    python a3.py cvrp-example
    python a3.py rushhour-hard-summary
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def run(command_args):
    """Run a delegated command from the repo root; return its exit code."""
    printable = " ".join(str(a) for a in command_args)
    print(f"$ {printable}", flush=True)
    return subprocess.run(command_args, cwd=REPO_ROOT).returncode


def python_script(*args):
    return [sys.executable] + [str(a) for a in args]


def cmd_sanity(_args):
    """The 80.64 sanity check: the multi-stage baseline on the tiny example
    instance must be feasible with cost about 80.64."""
    return run(python_script("scripts/run_cvrp_baseline.py",
                             "--instance", "examples/tiny_cvrp.vrp"))


def cmd_audit(_args):
    code = run(python_script("scripts/verify_report_matches_csv.py"))
    code |= run(python_script("scripts/audit_submission.py",
                              "--check-results", "--check-pdf"))
    return code


def cmd_test(_args):
    return run(python_script("-m", "pytest"))


def cmd_report_info(_args):
    pdf = REPO_ROOT / "report" / "assignment3_report.pdf"
    if pdf.exists():
        print(f"report PDF: {pdf.relative_to(REPO_ROOT)} "
              f"({pdf.stat().st_size} bytes)")
    else:
        print("report PDF: MISSING (run scripts/export_report_pdf.py)")
    for folder in ("report/evidence", "report/figures"):
        files = sorted((REPO_ROOT / folder).glob("*"))
        total = sum(f.stat().st_size for f in files if f.is_file())
        print(f"{folder}/: {len(files)} files, {total} bytes")
    return 0 if pdf.exists() else 1


def cmd_cvrp_example(_args):
    """A safe CVRP example that needs no official data; then show how to run
    the full final suite once the official .vrp files are in place."""
    code = run(python_script("scripts/run_cvrp_baseline.py",
                             "--instance", "examples/tiny_cvrp.vrp"))
    official = list((REPO_ROOT / "data" / "official_cvrp").glob("*.vrp"))
    print()
    if official:
        print(f"official instances found ({len(official)}); full suite command:")
    else:
        print("official .vrp files are user-provided and not committed —")
        print("place them under data/official_cvrp/ and run:")
    print("  python scripts/run_final_experiments.py "
          "--tuned-cvrp configs/tuned_cvrp_settings.json "
          "--rushhour-hard configs/rushhour_hard_benchmark.json")
    return code


def cmd_rushhour_hard_summary(_args):
    """Print the committed hard-benchmark summary (no rerun needed)."""
    summary = REPO_ROOT / "report" / "evidence" / "rushhour_hard_gp_gep_summary.csv"
    if not summary.exists():
        print(f"missing {summary.relative_to(REPO_ROOT)}; run:")
        print("  python scripts/run_gp_gep_hard_benchmark.py "
              "--puzzles examples/rushhour_hard_eval.txt --seeds 42 43 44")
        return 1
    print(f"# {summary.relative_to(REPO_ROOT)}")
    print(summary.read_text().rstrip())
    return 0


def cmd_rushhour_direct(_args):
    """Smoke-run the direct (no-A*) GP/GEP planner bonus; results local-only.
    Writes to a separate smoke dir so it never overwrites the real benchmark
    outputs under results/rushhour_direct/."""
    return run(python_script("scripts/run_gp_gep_direct_planner.py", "--smoke",
                             "--output-dir", "results/rushhour_direct_smoke"))


COMMANDS = {
    "sanity": (cmd_sanity, "run the 80.64 CVRP sanity check (tiny example)"),
    "rushhour-direct": (cmd_rushhour_direct,
                        "smoke-run the direct no-A* GP/GEP planner bonus"),
    "audit": (cmd_audit, "run the consistency gate and the submission audit"),
    "test": (cmd_test, "run the pytest suite"),
    "report-info": (cmd_report_info, "print report PDF/evidence/figure info"),
    "cvrp-example": (cmd_cvrp_example,
                     "run a safe CVRP example and show the full-suite command"),
    "rushhour-hard-summary": (cmd_rushhour_hard_summary,
                              "print the hard Rush Hour benchmark summary"),
}


def main():
    parser = argparse.ArgumentParser(
        prog="a3", description="Assignment 3 unified entry point.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, (_func, help_text) in COMMANDS.items():
        sub.add_parser(name, help=help_text)
    args = parser.parse_args()
    sys.exit(COMMANDS[args.command][0](args))


if __name__ == "__main__":
    main()
