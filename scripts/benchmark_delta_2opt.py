"""Microbenchmark for the Stage 11-A delta-evaluation claims in report §5.

Measures, on seeded random routes of several lengths:
1. one full best-improvement 2-opt scan that prices every candidate route
   with a complete route-cost recomputation (the pre-Stage-11 approach,
   improve_route_2opt_full's inner scan), versus one O(1)-delta scan
   (two_opt_route_delta) — reporting the per-pass speedup; both scans are
   verified to pick the same move.
2. the same delta scan with the distance matrix as plain Python lists
   versus a NumPy ndarray, to document why the ndarray was rejected
   (scalar indexing inside Python loops).

The committed evidence snapshot lives at
report/evidence/delta_eval_benchmark.txt; regenerate it with:

    .venv/bin/python scripts/benchmark_delta_2opt.py \
        --output report/evidence/delta_eval_benchmark.txt
"""

import argparse
import sys
import time
from pathlib import Path

# make "src" importable when the script is run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.cvrp.cost import route_cost
from src.cvrp.local_search import two_opt_route, two_opt_route_delta


def build_route_and_matrix(length, rng):
    """A seeded random depot-to-depot route over `length` customers with a
    plain-list Euclidean distance matrix (node 0 is the depot)."""
    coords = rng.uniform(0.0, 1000.0, size=(length + 1, 2))
    matrix = [[float(np.hypot(*(coords[i] - coords[j])))
               for j in range(length + 1)] for i in range(length + 1)]
    route = [0] + list(rng.permutation(range(1, length + 1))) + [0]
    return [int(n) for n in route], matrix


def full_scan_best_move(route, matrix):
    """One full best-improvement 2-opt scan with complete route-cost
    recomputation per candidate (the pre-Stage-11 pricing)."""
    base = route_cost(route, matrix)
    best_delta, best_candidate = -1e-9, None
    for candidate in two_opt_route(route, matrix):
        delta = route_cost(candidate, matrix) - base
        if delta < best_delta:
            best_delta, best_candidate = delta, candidate
    return best_delta, best_candidate


def best_of(callable_, repeats):
    """Minimum wall-clock time over `repeats` calls (stable microbenchmark)."""
    best = None
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = callable_()
        elapsed = time.perf_counter() - start
        best = elapsed if best is None else min(best, elapsed)
    return best, result


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark O(1)-delta vs full-recomputation 2-opt scans.")
    parser.add_argument("--lengths", type=int, nargs="+",
                        default=[10, 25, 50, 100, 200])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", help="optional path for the report file")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    lines = [
        "Delta-evaluation microbenchmark (scripts/benchmark_delta_2opt.py)",
        f"seed {args.seed}, best of {args.repeats} repeats per timing, "
        "one best-improvement 2-opt scan per measurement",
        "",
        "part 1: full route-cost recomputation vs O(1) delta scan",
        "route_len  full_scan_ms  delta_scan_ms  speedup",
    ]
    speedups = []
    for length in args.lengths:
        route, matrix = build_route_and_matrix(length, rng)
        t_full, (full_delta, full_candidate) = best_of(
            lambda: full_scan_best_move(route, matrix), args.repeats)
        t_delta, move = best_of(
            lambda: two_opt_route_delta(route, matrix), args.repeats)
        # both scans must select the same move (same first-best tie order)
        if (move is None) != (full_candidate is None):
            sys.exit(f"scan mismatch at length {length}: one found no move")
        if move is not None:
            delta, i, j = move
            applied = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
            if applied != full_candidate or abs(delta - full_delta) > 1e-6:
                sys.exit(f"scan mismatch at length {length}: different move")
        speedup = t_full / t_delta
        speedups.append(speedup)
        lines.append(f"{length:9d}  {1000 * t_full:12.3f}  "
                     f"{1000 * t_delta:13.3f}  {speedup:6.1f}x")

    lines += ["", f"speedup range over route lengths "
              f"{min(args.lengths)}-{max(args.lengths)}: "
              f"{min(speedups):.0f}x to {max(speedups):.0f}x", "",
              "part 2: delta scan, plain-list matrix vs NumPy ndarray "
              "(scalar indexing in Python loops)",
              "route_len  list_ms  ndarray_ms  ndarray_slowdown"]
    slowdowns = []
    for length in args.lengths[-2:]:
        route, matrix = build_route_and_matrix(length, rng)
        array = np.array(matrix)
        t_list, _ = best_of(
            lambda: two_opt_route_delta(route, matrix), args.repeats)
        t_array, _ = best_of(
            lambda: two_opt_route_delta(route, array), args.repeats)
        slowdown = t_array / t_list
        slowdowns.append(slowdown)
        lines.append(f"{length:9d}  {1000 * t_list:7.3f}  "
                     f"{1000 * t_array:10.3f}  {slowdown:15.1f}x")
    lines += ["", f"ndarray slowdown range: {min(slowdowns):.1f}x to "
              f"{max(slowdowns):.1f}x", ""]

    text = "\n".join(lines)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
        print(f"written to: {output}")


if __name__ == "__main__":
    main()
