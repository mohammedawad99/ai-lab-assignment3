"""Strict gate: the report's CVRP headline claims must match the committed
evidence CSVs exactly (tolerance 0.01 on printed percentages).

Checks, all against report/evidence/:
1. Per-instance headline best gaps in the report equal the minimum
   best_gap_percent in cvrp_all_policy_effective_summary.csv (canonical
   3-seed final run) and agree with cvrp_algorithm_mean_gaps.csv.
2. Per-algorithm mean gaps in the report equal cvrp_algorithm_mean_gaps.csv.
3. B&B/LDS: 2-decimal mean and the beats-baseline count in the report match
   the evidence.
4. Seed policy: the canonical evidence must contain exactly seeds 42/43/44,
   the report must declare them, and any 8-seed mention must stay inside a
   labeled robustness context (the report must carry the canonical-source
   sentence).

Exit code 0 when consistent, 1 with a message per failure otherwise.

Usage:
    .venv/bin/python scripts/verify_report_matches_csv.py
"""

import csv
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = REPO_ROOT / "report" / "evidence"
REPORT = REPO_ROOT / "report" / "assignment3_report.md"
INSTANCES = ["P-n16-k8", "E-n22-k4", "A-n32-k5", "A-n80-k10",
             "X-n101-k25", "M-n200-k17"]
TOLERANCE = 0.01


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def number_in_text(text, value, decimals=(4, 2)):
    """True if the value appears in the text at any of the given precisions."""
    return any(f"{value:.{d}f}" in text for d in decimals)


def main():
    failures = []

    def check(name, ok, detail=""):
        print(f"{'PASS' if ok else 'FAIL'}  {name}"
              + (f" ({detail})" if detail else ""))
        if not ok:
            failures.append(name)

    text = REPORT.read_text(encoding="utf-8")
    summary = read_rows(EVIDENCE / "cvrp_all_summary.csv")
    effective = read_rows(EVIDENCE / "cvrp_all_policy_effective_summary.csv")
    mean_gaps = read_rows(EVIDENCE / "cvrp_algorithm_mean_gaps.csv")
    by_algo = {r["algorithm"]: r for r in mean_gaps}

    # 1. headline best gap per instance
    for inst in INSTANCES:
        best_eff = min(float(r["best_gap_percent"]) for r in effective
                       if r["instance"] == inst)
        best_mg = min(float(r[inst]) for r in mean_gaps)
        check(f"evidence CSVs agree on best gap for {inst}",
              abs(best_eff - best_mg) <= TOLERANCE,
              f"effective {best_eff:.4f} vs mean_gaps {best_mg:.4f}")
        check(f"report states best gap for {inst}",
              number_in_text(text, best_eff), f"{best_eff:.4f}")

    # 2. per-algorithm mean gaps
    for row in mean_gaps:
        mean = float(row["mean_best_gap_percent"])
        check(f"report states mean gap for {row['algorithm']}",
              number_in_text(text, mean, decimals=(3, 2)), f"{mean:.3f}")

    # 3. B&B/LDS specifics
    bnb, base = by_algo["cvrp_bnb_lds"], by_algo["baseline"]
    bnb_mean = sum(float(bnb[i]) for i in INSTANCES) / len(INSTANCES)
    check("report states B&B 2-decimal mean",
          f"{bnb_mean:.2f}%" in text, f"{bnb_mean:.2f}%")
    beats = sum(1 for i in INSTANCES
                if float(bnb[i]) < float(base[i]) - 1e-9)
    check("report states B&B beats-baseline count",
          f"{beats}/6" in text, f"{beats}/6")

    # 4. seed policy: canonical evidence is exactly seeds 42-44 and the
    # report declares the canonical 3-seed source
    seeds = {r["best_seed"] for r in summary if r.get("best_seed")}
    check("canonical evidence uses only seeds 42/43/44",
          seeds <= {"42", "43", "44"}, ",".join(sorted(seeds)))
    check("report declares seeds 42, 43 and 44",
          "seeds 42, 43 and 44" in text)
    check("report declares the canonical 3-seed source for headline tables",
          "canonical source of every headline table" in text)
    if "42–49" in text or "42-49" in text:
        check("8-seed mentions are labeled as robustness",
              "robustness" in text and "not the source of any" in text)

    # 5. explicit ILS (Stage 13-A): the report's ILS numbers must equal the
    # committed ILS evidence, and every ILS run must be feasible
    ils_path = EVIDENCE / "cvrp_ils_summary.csv"
    check("ILS evidence snapshot exists", ils_path.exists())
    if ils_path.exists():
        ils = {r["instance"]: r for r in read_rows(ils_path)}
        check("ILS evidence covers all six instances",
              set(ils) == set(INSTANCES), ",".join(sorted(ils)))
        check("all ILS evidence runs are feasible",
              all(r["runs"] == r["feasible_runs"] for r in ils.values()))
        for inst in INSTANCES:
            gap = float(ils[inst]["best_gap_percent"])
            check(f"report states ILS best gap for {inst}",
                  number_in_text(text, gap), f"{gap:.4f}")
        ils_mean = sum(float(r["best_gap_percent"])
                       for r in ils.values()) / len(ils)
        check("report states ILS mean best gap",
              number_in_text(text, ils_mean, decimals=(3, 2)),
              f"{ils_mean:.3f}")

    # 6. per-instance mean +/- std table (Section 9) must match the summary
    # CSV cell by cell, in the exact "mean±std" one-decimal format it uses
    missing_cells = []
    for row in summary:
        cell = (f"{float(row['mean_cost']):.1f}"
                f"±{float(row['std_cost']):.1f}")
        if cell not in text:
            missing_cells.append(f"{row['instance']}/{row['algorithm']}")
    check("report per-instance mean±std cells match cvrp_all_summary",
          not missing_cells, ", ".join(missing_cells[:4]))

    # 7. the Section 5 delta-evaluation performance claims must match the
    # committed microbenchmark evidence (both measured ranges, verbatim)
    bench_path = EVIDENCE / "delta_eval_benchmark.txt"
    check("delta-eval benchmark evidence exists", bench_path.exists())
    if bench_path.exists():
        bench = bench_path.read_text(encoding="utf-8")
        speed = re.search(
            r"speedup range over route lengths [\d-]+: (\d+)x to (\d+)x", bench)
        slow = re.search(
            r"ndarray slowdown range: ([\d.]+)x to ([\d.]+)x", bench)
        check("benchmark evidence has both measured ranges",
              bool(speed and slow))
        if speed:
            claim = f"{speed.group(1)}–{speed.group(2)}×"
            check("report states the measured delta-eval speedup range",
                  claim in text, claim)
        if slow:
            claim = f"{slow.group(1)}–{slow.group(2)}×"
            check("report states the measured ndarray slowdown range",
                  claim in text, claim)

    print(f"\nreport-vs-evidence: {'PASS' if not failures else 'FAIL'} "
          f"({len(failures)} failed check(s))")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
