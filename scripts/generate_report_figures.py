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
    RESULTS / "rushhour_hard" / "gp_gep_hard_summary.csv",
    RESULTS / "rushhour_hard" / "manual_heuristics_eval.csv",
    RESULTS / "final_execution_manifest.json",
]

# pre-tuning best gaps (previous final run, recorded in Stage 10-A snapshot
# and in report/evidence/final_v2_summary.txt) for the before/after figure
PREVIOUS_BEST_GAPS = {
    "P-n16-k8": 0.4327, "E-n22-k4": 0.0746, "A-n32-k5": 0.3931,
    "A-n80-k10": 4.0276, "X-n101-k25": 25.4508, "M-n200-k17": 5.9946,
}


def apply_alns_policy(rows):
    """Return the policy-effective row view: the assignment 'alns' result is
    the enhanced variant except on the instances excluded by the
    pre-declared rule in configs/tuned_cvrp_settings.json."""
    import json
    policy = json.loads((REPO_ROOT / "configs" / "tuned_cvrp_settings.json")
                        .read_text()).get("alns_policy", {})
    excluded = set(policy.get("enhanced_excluded_instances", []))
    effective = []
    for r in rows:
        algo, inst = r["algorithm"], r["instance"]
        if algo == "cvrp_alns" and inst not in excluded:
            continue
        if algo == "cvrp_alns_enhanced":
            if inst in excluded:
                continue
            r = dict(r, algorithm="cvrp_alns")
        effective.append(r)
    return effective

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

    # ---- CVRP charts (policy-effective view: alns = enhanced except M) ----
    rows = apply_alns_policy(read_rows(RESULTS / "raw" / "cvrp_all_instances.csv"))
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

    # ---- before/after tuning comparison ----
    new_best = {name: min(float(r["gap_percent"]) for r in feasible
                          if r["instance"] == name) for name in INSTANCES}
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    width = 0.38
    xs = range(len(INSTANCES))
    old_bars = ax.bar([x - width / 2 for x in xs],
                      [PREVIOUS_BEST_GAPS[i] for i in INSTANCES], width,
                      label="before tuning")
    new_bars = ax.bar([x + width / 2 for x in xs],
                      [new_best[i] for i in INSTANCES], width,
                      label="after tuning (policy)")
    for bars in (old_bars, new_bars):
        for bar in bars:
            ax.annotate(f"{bar.get_height():.2f}",
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha="center", va="bottom", fontsize=7)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(INSTANCES, rotation=30, ha="right")
    ax.set_ylabel("best gap vs BKS (%)")
    ax.set_title("CVRP best gaps before vs after Stage 10 tuning "
                 "(P/A80/X improved, no regressions)")
    ax.legend()
    fig.tight_layout()
    path = FIGURES / "cvrp_before_after_tuning.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    created.append(path)

    # ---- Rush Hour hard benchmark charts ----
    hard = read_rows(RESULTS / "rushhour_hard" / "gp_gep_hard_summary.csv")
    manual = [r for r in hard if r["name"].startswith("manual_")]
    evolved = [r for r in hard if not r["name"].startswith("manual_")]

    ladder = sorted(manual, key=lambda r: float(r["eval_fitness"]))
    save_bar([r["name"].replace("manual_", "") for r in ladder],
             [float(r["expanded"]) for r in ladder],
             "Rush Hour hard set: manual heuristic ladder (A* expansions, lower is better)",
             "expanded nodes", "rushhour_manual_heuristic_ladder.png",
             value_fmt="{:.0f}")

    best_manual = max(manual, key=lambda r: float(r["eval_fitness"]))
    best_gp = max((r for r in evolved if r["name"].startswith("gp_")),
                  key=lambda r: float(r["eval_fitness"]))
    best_gep = max((r for r in evolved if r["name"].startswith("gep_")),
                   key=lambda r: float(r["eval_fitness"]))
    trio = [best_manual, best_gp, best_gep]
    save_bar([f"{r['name']}\n({r['solved']}/{r['puzzles']} solved)" for r in trio],
             [float(r["expanded"]) for r in trio],
             "Best manual vs best GP vs best GEP (A* expansions on hard set)",
             "expanded nodes", "rushhour_gp_gep_vs_manual.png",
             value_fmt="{:.0f}")

    save_bar([r["name"].replace("_rushhour", "") for r in evolved],
             [float(r["eval_fitness"]) for r in evolved],
             "GP/GEP fitness per seed: variance dominates the method difference",
             "eval fitness", "rushhour_gp_gep_seed_variance.png",
             value_fmt="{:.0f}")

    # per-puzzle difficulty under the best manual guide
    per_puzzle = read_rows(RESULTS / "rushhour_hard" / "manual_heuristics_eval.csv")
    depth_rows = [r for r in per_puzzle if r["heuristic_name"] == "blocker_depth"]
    depth_rows.sort(key=lambda r: int(r["puzzle_id"]))
    save_bar([f"p{int(r['puzzle_id']) + 1}" for r in depth_rows],
             [max(1.0, float(r["expanded_nodes"])) for r in depth_rows],
             "Hard set per-puzzle A* expansions (blocker_depth heuristic, log scale)",
             "expanded nodes", "rushhour_per_puzzle_difficulty.png",
             value_fmt="{:.0f}", log=True)

    # ---- seed-robustness boxplots (only when the local run exists) ----
    robustness = REPO_ROOT / "results" / "robustness" / "cvrp_seed_robustness_raw.csv"
    if robustness.exists():
        rob = [r for r in read_rows(robustness) if r["gap_percent"]]
        rob_instances = [i for i in INSTANCES
                         if any(r["instance"] == i for r in rob)]
        algorithms_order = ["baseline", "bnb_lds", "sa", "tabu", "aco",
                            "ga_island", "alns_effective"]

        def grouped(metric):
            data = {}
            for inst in rob_instances:
                data[inst] = [
                    [float(r[metric]) for r in rob
                     if r["instance"] == inst and r["algorithm"] == algo]
                    for algo in algorithms_order
                ]
            return data

        for metric, title, ylabel, filename, use_log in [
            ("gap_percent", "CVRP gap distribution over seeds",
             "gap vs BKS (%)", "cvrp_seed_gap_boxplots.png", False),
            ("elapsed_time", "CVRP runtime distribution over seeds (log scale)",
             "seconds", "cvrp_seed_runtime_boxplots.png", True),
        ]:
            data = grouped(metric)
            fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6))
            for ax, inst in zip(axes.flat, rob_instances):
                ax.boxplot(data[inst], tick_labels=[a.replace("_effective", "")
                                                    for a in algorithms_order])
                ax.set_title(inst, fontsize=10)
                ax.tick_params(axis="x", rotation=45, labelsize=7)
                ax.tick_params(axis="y", labelsize=8)
                if use_log:
                    ax.set_yscale("log")
            fig.suptitle(f"{title} ({len({r['seed'] for r in rob})} seeds per box)")
            fig.supylabel(ylabel)
            fig.tight_layout()
            path = FIGURES / filename
            fig.savefig(path, dpi=150)
            plt.close(fig)
            created.append(path)

        # win rate vs baseline per instance/algorithm
        fig, ax = plt.subplots(figsize=(9.0, 4.4))
        width = 0.8 / (len(algorithms_order) - 1)
        compare = [a for a in algorithms_order if a != "baseline"]
        for k, algo in enumerate(compare):
            rates = []
            for inst in rob_instances:
                base = {r["seed"]: float(r["gap_percent"]) for r in rob
                        if r["instance"] == inst and r["algorithm"] == "baseline"}
                mine = {r["seed"]: float(r["gap_percent"]) for r in rob
                        if r["instance"] == inst and r["algorithm"] == algo}
                seeds = sorted(set(base) & set(mine))
                wins = sum(1 for s in seeds if mine[s] < base[s] - 1e-9)
                rates.append(100.0 * wins / len(seeds) if seeds else 0.0)
            positions = [i + k * width for i in range(len(rob_instances))]
            ax.bar(positions, rates, width=width,
                   label=algo.replace("_effective", ""))
        centers = [i + 0.4 - width / 2 for i in range(len(rob_instances))]
        ax.set_xticks(centers)
        ax.set_xticklabels(rob_instances, rotation=15)
        ax.set_ylabel("% of seeds beating baseline")
        ax.set_title("Per-seed win rate against the multi-stage baseline")
        ax.legend(fontsize=7, ncol=3)
        fig.tight_layout()
        path = FIGURES / "cvrp_seed_win_rate_vs_baseline.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        created.append(path)
    else:
        print("note: results/robustness raw CSV not found, "
              "seed boxplot figures not regenerated (committed PNGs kept)")

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
