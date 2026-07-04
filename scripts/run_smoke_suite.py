"""Fast end-to-end smoke suite: all CVRP and Ackley algorithms on tiny settings.

This is NOT a final experiment — just a quick check that every algorithm
runs, produces feasible results, and can be summarized.

Usage:
    python scripts/run_smoke_suite.py --output-dir results/smoke_suite
"""

import argparse
import sys
from pathlib import Path

# make "src" importable when the script is run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments.ackley_experiments import ACKLEY_FIELDNAMES, run_ackley_experiments
from src.experiments.csv_utils import write_dict_rows
from src.experiments.cvrp_experiments import CVRP_FIELDNAMES, run_cvrp_experiments
from src.experiments.presets import (
    SMOKE_ACKLEY_ALGORITHMS,
    SMOKE_ACKLEY_DIMENSION,
    SMOKE_BUDGET,
    SMOKE_CVRP_ALGORITHMS,
    SMOKE_SEEDS,
    SMOKE_TIMEOUT_SEC,
)
from src.experiments.summary import (
    ACKLEY_SUMMARY_FIELDNAMES,
    CVRP_SUMMARY_FIELDNAMES,
    safe_float,
    summarize_ackley_rows,
    summarize_cvrp_rows,
    write_summary_csv,
)

KNOWN_ALGORITHMS = set(SMOKE_CVRP_ALGORITHMS)


def fail(message):
    print(message, file=sys.stderr)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Run the smoke experiment suite.")
    parser.add_argument("--cvrp-instance", default="examples/tiny_cvrp.vrp")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SMOKE_SEEDS))
    parser.add_argument("--budget", type=int, default=SMOKE_BUDGET)
    parser.add_argument("--timeout", type=float, default=SMOKE_TIMEOUT_SEC)
    parser.add_argument("--ackley-dimension", type=int, default=SMOKE_ACKLEY_DIMENSION)
    parser.add_argument("--cvrp-algorithms", nargs="+", default=list(SMOKE_CVRP_ALGORITHMS))
    parser.add_argument("--ackley-algorithms", nargs="+", default=list(SMOKE_ACKLEY_ALGORITHMS))
    parser.add_argument("--include-baseline", action="store_true")
    parser.add_argument("--include-random-search", action="store_true")
    args = parser.parse_args()

    for name in args.cvrp_algorithms + args.ackley_algorithms:
        if name not in KNOWN_ALGORITHMS:
            fail(f"unknown algorithm '{name}', known: {', '.join(sorted(KNOWN_ALGORITHMS))}")

    output_dir = Path(args.output_dir)
    cvrp_algorithms = (["baseline"] if args.include_baseline else []) + args.cvrp_algorithms
    ackley_algorithms = (["random_search"] if args.include_random_search else []) + args.ackley_algorithms

    # CVRP smoke runs (also writes solution files under output_dir/cvrp_solutions)
    try:
        cvrp_rows = run_cvrp_experiments(
            [args.cvrp_instance], cvrp_algorithms, args.seeds,
            budget=args.budget, timeout_sec=args.timeout, output_dir=output_dir,
        )
    except Exception as exc:
        fail(f"could not run CVRP smoke on '{args.cvrp_instance}': {exc}")
    cvrp_raw = output_dir / "raw" / "cvrp_results.csv"
    write_dict_rows(cvrp_raw, cvrp_rows, CVRP_FIELDNAMES)

    # Ackley smoke runs
    ackley_rows = run_ackley_experiments(
        ackley_algorithms, args.seeds, budget=args.budget,
        timeout_sec=args.timeout, dimension=args.ackley_dimension,
    )
    ackley_raw = output_dir / "raw" / "ackley_results.csv"
    write_dict_rows(ackley_raw, ackley_rows, ACKLEY_FIELDNAMES)

    # summaries
    cvrp_summary = summarize_cvrp_rows(cvrp_rows)
    cvrp_summary_path = output_dir / "summary" / "cvrp_summary.csv"
    write_summary_csv(cvrp_summary_path, cvrp_summary, CVRP_SUMMARY_FIELDNAMES)

    ackley_summary = summarize_ackley_rows(ackley_rows)
    ackley_summary_path = output_dir / "summary" / "ackley_summary.csv"
    write_summary_csv(ackley_summary_path, ackley_summary, ACKLEY_SUMMARY_FIELDNAMES)

    best_cost = min((c for c in (safe_float(r["best_cost"]) for r in cvrp_rows)
                     if c is not None), default=None)
    best_value = min((v for v in (safe_float(r["best_value"]) for r in ackley_rows)
                      if v is not None), default=None)

    print(f"cvrp raw rows: {len(cvrp_rows)} -> {cvrp_raw}")
    print(f"ackley raw rows: {len(ackley_rows)} -> {ackley_raw}")
    print(f"cvrp summary rows: {len(cvrp_summary)} -> {cvrp_summary_path}")
    print(f"ackley summary rows: {len(ackley_summary)} -> {ackley_summary_path}")
    print(f"best cvrp cost: {best_cost if best_cost is not None else '(none)'}")
    print(f"best ackley value: {best_value if best_value is not None else '(none)'}")
    print(f"cvrp solutions dir: {output_dir / 'cvrp_solutions'}")
    sys.exit(0)


if __name__ == "__main__":
    main()
