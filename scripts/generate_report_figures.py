"""Generate the committed report figures under report/figures/.

Everything is built locally from real repository files:
- charts from the final result CSVs under results/final_experiments/
- code snippet images rendered from the actual source files
- terminal images rendered from real command output (audit + report numbers)

Usage:
    python scripts/generate_report_figures.py
"""

import csv
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results" / "final_experiments"
FIGURES = REPO_ROOT / "report" / "figures"

INSTANCES = ["P-n16-k8", "E-n22-k4", "A-n32-k5", "A-n80-k10", "X-n101-k25", "M-n200-k17"]

REQUIRED_RESULTS = [
    RESULTS / "raw" / "cvrp_all_instances.csv",
    RESULTS / "summary" / "cvrp_all_summary.csv",
    RESULTS / "raw" / "ackley_d10.csv",
    RESULTS / "summary" / "ackley_d10_summary.csv",
    RESULTS / "raw" / "gp_gep_comparison_runs.csv",
    RESULTS / "final_execution_manifest.json",
]

created = []


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def save_bar(labels, values, title, ylabel, filename, value_fmt="{:.2f}", log=False):
    fig, ax = plt.subplots(figsize=(max(6.5, 1.05 * len(labels)), 4.2))
    bars = ax.bar(range(len(labels)), values)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")
    for bar, value in zip(bars, values):
        ax.annotate(value_fmt.format(value), (bar.get_x() + bar.get_width() / 2,
                                              bar.get_height()),
                    ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    path = FIGURES / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    created.append(path)


def render_text_image(lines, filename, dark=True, fontsize=7.5, title=None):
    """Monospace text rendered as an image (code / terminal style)."""
    lines = [line[:120] for line in lines]
    if title:
        lines = [title, ""] + lines
    height = max(1.6, 0.155 * len(lines) + 0.4)
    width = min(11.5, max(7.0, 0.062 * max(len(l) for l in lines) + 0.6))
    bg, fg = ("#1e1e1e", "#d8d8d8") if dark else ("#f8f8f8", "#111111")
    fig = plt.figure(figsize=(width, height))
    fig.patch.set_facecolor(bg)
    y = 1.0 - 0.22 / height
    for line in lines:
        fig.text(0.015, y, line, fontsize=fontsize, family="monospace",
                 color=fg, va="top")
        y -= 0.155 / height
    path = FIGURES / filename
    fig.savefig(path, dpi=150, facecolor=bg)
    plt.close(fig)
    created.append(path)


def extract_snippet(path, marker, max_lines=30):
    lines = (REPO_ROOT / path).read_text().splitlines()
    for i, line in enumerate(lines):
        if marker in line:
            chunk = lines[i:i + max_lines]
            return [f"{i + 1 + j:4d} | {text}" for j, text in enumerate(chunk)]
    raise SystemExit(f"marker '{marker}' not found in {path}")


def main():
    missing = [str(p) for p in REQUIRED_RESULTS if not p.exists()]
    if missing:
        print("missing required result files:", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        sys.exit(1)
    FIGURES.mkdir(parents=True, exist_ok=True)

    # ---- CVRP charts ----
    rows = read_rows(RESULTS / "raw" / "cvrp_all_instances.csv")
    feasible = [r for r in rows if r["feasible"] == "True" and r["gap_percent"]]

    best_gaps = [min(float(r["gap_percent"]) for r in feasible
                     if r["instance"] == name) for name in INSTANCES]
    save_bar(INSTANCES, best_gaps, "CVRP: best feasible gap per instance",
             "gap vs BKS (%)", "cvrp_best_gap_by_instance.png")

    algorithms = sorted({r["algorithm"] for r in feasible})
    mean_gaps = [sum(float(r["gap_percent"]) for r in feasible if r["algorithm"] == a)
                 / len([r for r in feasible if r["algorithm"] == a])
                 for a in algorithms]
    save_bar([a.replace("cvrp_", "") for a in algorithms], mean_gaps,
             "CVRP: mean gap per algorithm (all runs, all instances)",
             "mean gap vs BKS (%)", "cvrp_mean_gap_by_algorithm.png")

    mean_elapsed = [sum(float(r["elapsed_time"]) for r in rows if r["instance"] == name)
                    / len([r for r in rows if r["instance"] == name])
                    for name in INSTANCES]
    save_bar(INSTANCES, mean_elapsed,
             "CVRP: mean runtime per run, by instance (log scale)",
             "seconds", "cvrp_runtime_by_instance.png", log=True)

    # ---- Ackley charts ----
    summary = read_rows(RESULTS / "summary" / "ackley_d10_summary.csv")
    summary.sort(key=lambda r: float(r["best_value"]))
    names = [r["algorithm"].replace("ackley_", "") for r in summary]
    save_bar(names, [float(r["best_value"]) for r in summary],
             "Ackley d=10: best value per algorithm (lower is better)",
             "best f(x)", "ackley_best_value_by_algorithm.png",
             value_fmt="{:.3f}")
    save_bar(names, [float(r["mean_elapsed_time"]) for r in summary],
             "Ackley d=10: mean runtime per run (log scale)",
             "seconds", "ackley_runtime_by_algorithm.png",
             value_fmt="{:.3f}", log=True)

    # ---- GP/GEP charts ----
    gp_rows = read_rows(RESULTS / "raw" / "gp_gep_comparison_runs.csv")
    labels = [f"{r['algorithm'].replace('_rushhour', '')}\nseed {r['seed']}"
              for r in gp_rows]
    save_bar(labels, [float(r["eval_fitness"]) for r in gp_rows],
             "GP vs GEP: eval fitness per run", "fitness",
             "gp_gep_eval_fitness.png", value_fmt="{:.0f}")
    save_bar(labels, [float(r["eval_total_expanded_nodes"]) for r in gp_rows],
             "GP vs GEP: A* expanded nodes on eval set", "expanded nodes",
             "gp_gep_expanded_nodes.png", value_fmt="{:.0f}")

    # ---- code snippet figures (real source lines) ----
    render_text_image(extract_snippet("src/cvrp/local_search.py",
                                      "def build_routes_subset_sum_packing", 28),
                      "code_baseline_repair.png",
                      title="src/cvrp/local_search.py — subset-sum vehicle repair")
    render_text_image(extract_snippet("src/gep/decoder.py",
                                      "def decode_genome_to_tree", 27),
                      "code_gep_decoder.png",
                      title="src/gep/decoder.py — Karva decoding")
    render_text_image(extract_snippet("src/experiments/final_execution.py",
                                      "def run_final_experiment_suite", 30),
                      "code_final_runner.png",
                      title="src/experiments/final_execution.py — final suite")

    # ---- terminal figures (real command output) ----
    audit = subprocess.run(
        [sys.executable, "scripts/audit_submission.py", "--check-results", "--check-pdf"],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    render_text_image(
        ["$ python scripts/audit_submission.py --check-results --check-pdf", ""]
        + audit.stdout.splitlines(),
        "terminal_audit_pass.png", fontsize=7)

    numbers = subprocess.run(
        [sys.executable, "scripts/extract_report_numbers.py"],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    render_text_image(
        ["$ python scripts/extract_report_numbers.py", ""]
        + numbers.stdout.splitlines(),
        "terminal_final_rows.png", fontsize=7)

    for path in created:
        print(f"generated: {path.relative_to(REPO_ROOT)}")
    print(f"total figures: {len(created)}")
    sys.exit(0)


if __name__ == "__main__":
    main()
