"""Train/evaluate the direct (no-A*) GP and GEP Rush Hour planners.

Both frameworks evolve a move-scoring policy over the direct move features
and are rolled out greedily (src/rushhour/direct_planner.py) — A* is never
called. Three simple direct baselines run under the identical rollout for
an honest comparison. Outputs are local-only under --output-dir.

Usage:
    .venv/bin/python scripts/run_gp_gep_direct_planner.py \
        --train-puzzles examples/rushhour_hard_eval.txt \
        --eval-puzzles examples/rushhour_hard_eval.txt \
        --seeds 42 43 44 --population 30 --generations 20 \
        --max-steps 120 --timeout-per-puzzle 2 \
        --output-dir results/rushhour_direct
    .venv/bin/python scripts/run_gp_gep_direct_planner.py --smoke
"""

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from src.rushhour.direct_fitness import (
    evaluate_direct_policy,
    policy_from_tree,
    run_gep_direct,
    run_gp_direct,
)
from src.rushhour.direct_planner import (
    greedy_blocker_depth_policy,
    greedy_red_distance_policy,
    random_policy,
)
from src.rushhour.gp_gep_comparison import expression_diversity
from src.rushhour.puzzle_sets import load_puzzle_set

RUN_FIELDS = ["method", "seed", "train_solved", "train_puzzles", "eval_solved",
              "eval_puzzles", "mean_steps_solved", "total_repeated_states",
              "runtime_seconds", "best_fitness", "expression_size",
              "diversity", "best_expression", "error"]


def eval_row(method, seed, evaluation, extra=None):
    solved_steps = [r.steps for r in evaluation.results if r.solved]
    row = {
        "method": method, "seed": seed,
        "train_solved": "", "train_puzzles": "",
        "eval_solved": evaluation.solved_count,
        "eval_puzzles": evaluation.puzzle_count,
        "mean_steps_solved": (f"{statistics.mean(solved_steps):.1f}"
                              if solved_steps else ""),
        "total_repeated_states": evaluation.total_repeated_states,
        "runtime_seconds": f"{evaluation.total_runtime:.2f}",
        "best_fitness": "", "expression_size": "", "diversity": "",
        "best_expression": "", "error": "",
    }
    row.update(extra or {})
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train-puzzles", default="examples/rushhour_hard_eval.txt")
    parser.add_argument("--eval-puzzles", default="examples/rushhour_hard_eval.txt")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--population", type=int, default=30)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--timeout-per-puzzle", type=float, default=2.0)
    parser.add_argument("--budget-per-individual", type=float, default=8.0)
    parser.add_argument("--output-dir", default="results/rushhour_direct")
    parser.add_argument("--smoke", action="store_true",
                        help="tiny run (2 puzzles, small population) for tests")
    args = parser.parse_args()

    train = load_puzzle_set(REPO_ROOT / args.train_puzzles)
    evaluation_set = load_puzzle_set(REPO_ROOT / args.eval_puzzles)
    if args.smoke:
        train, evaluation_set = train[:2], evaluation_set[:2]
        args.seeds = args.seeds[:1]
        args.population, args.generations = 4, 1
        args.max_steps = min(args.max_steps, 40)

    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rollout = dict(max_steps=args.max_steps,
                   timeout_per_puzzle=args.timeout_per_puzzle)

    # ---- direct manual baselines (same rollout mechanism, no A*) ----
    baseline_rows = []
    for name, policy in [
        ("random_legal_move", random_policy(np.random.default_rng(42))),
        ("greedy_blocking_depth_move", greedy_blocker_depth_policy),
        ("greedy_red_distance_move", greedy_red_distance_policy),
    ]:
        evaluation = evaluate_direct_policy(policy, evaluation_set, **rollout)
        baseline_rows.append(eval_row(name, 42, evaluation))
        print(f"{name:28s} solved {evaluation.solved_count}/"
              f"{evaluation.puzzle_count}  repeated "
              f"{evaluation.total_repeated_states}  "
              f"{evaluation.total_runtime:.1f}s")

    # ---- GP / GEP direct training + evaluation ----
    run_rows = []
    for algorithm, runner in (("gp_direct", run_gp_direct),
                              ("gep_direct", run_gep_direct)):
        for seed in args.seeds:
            started = time.perf_counter()
            result = runner(train, generations=args.generations,
                            population_size=args.population, seed=seed,
                            max_steps=args.max_steps,
                            timeout_per_puzzle=args.timeout_per_puzzle,
                            budget_per_individual=args.budget_per_individual)
            evaluation = evaluate_direct_policy(
                policy_from_tree(result.best.tree), evaluation_set, **rollout)
            row = eval_row(algorithm, seed, evaluation, extra={
                "train_solved": result.best.solved_count,
                "train_puzzles": result.best.puzzle_count,
                "runtime_seconds": f"{time.perf_counter() - started:.2f}",
                "best_fitness": f"{result.best.fitness:.1f}",
                "expression_size": result.best.expression_size,
                "diversity": f"{expression_diversity(result.expressions):.2f}",
                "best_expression": result.best.expression[:120],
            })
            run_rows.append(row)
            print(f"{algorithm:10s} seed {seed}: train "
                  f"{result.best.solved_count}/{result.best.puzzle_count}  "
                  f"eval {evaluation.solved_count}/{evaluation.puzzle_count}  "
                  f"fitness {result.best.fitness:.0f}  "
                  f"{time.perf_counter() - started:.0f}s")

    # ---- write outputs ----
    def write_csv(name, rows):
        with open(output_dir / name, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=RUN_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    write_csv("direct_gp_gep_runs.csv", run_rows)
    write_csv("direct_manual_baselines.csv", baseline_rows)

    def summarize(rows, name):
        out = []
        for method in sorted({r["method"] for r in rows}):
            group = [r for r in rows if r["method"] == method]
            out.append({
                "method": method,
                "runs": len(group),
                "best_eval_solved": max(int(r["eval_solved"]) for r in group),
                "mean_eval_solved": f"{statistics.mean(int(r['eval_solved']) for r in group):.1f}",
                "eval_puzzles": group[0]["eval_puzzles"],
                "mean_runtime_seconds": f"{statistics.mean(float(r['runtime_seconds']) for r in group):.1f}",
            })
        with open(output_dir / name, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(out[0].keys()))
            writer.writeheader()
            writer.writerows(out)
        return out

    summary = summarize(run_rows, "direct_gp_gep_summary.csv")
    baseline_summary = summarize(baseline_rows, "direct_manual_summary.csv")

    manifest = {
        "train_puzzles": args.train_puzzles, "eval_puzzles": args.eval_puzzles,
        "train_count": len(train), "eval_count": len(evaluation_set),
        "seeds": args.seeds, "population": args.population,
        "generations": args.generations, "max_steps": args.max_steps,
        "timeout_per_puzzle": args.timeout_per_puzzle,
        "budget_per_individual": args.budget_per_individual,
        "smoke": args.smoke,
        "note": "direct greedy policy rollout, no A* anywhere in this benchmark",
    }
    (output_dir / "direct_planner_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")

    print("\nsummary (eval set):")
    for entry in summary + baseline_summary:
        print(f"  {entry['method']:28s} best {entry['best_eval_solved']}/"
              f"{entry['eval_puzzles']}  mean {entry['mean_eval_solved']}")
    print(f"outputs: {output_dir}")
    sys.exit(0)


if __name__ == "__main__":
    main()
