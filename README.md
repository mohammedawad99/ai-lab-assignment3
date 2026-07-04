# AI Lab — Assignment 3: CVRP + Generative Heuristics (GP/GEP)

Course 203.3630 — Artificial Intelligence Lab.

Two parts:
- **Part A — CVRP:** solve the Capacitated Vehicle Routing Problem and compare six
  search algorithms (TS, ACO, SA, GA with Island Model, ALNS, Branch & Bound/LDS),
  plus at least one multi-stage heuristic. An Ackley-function warm-up validates each solver.
- **Part B — Generative AI:** use GP and GEP to evolve improved Rush Hour heuristics,
  evaluated by running A* with the candidate heuristic.

## Status

Stage 3-A — Ackley warm-up framework done. No metaheuristic solvers or GP/GEP yet.

## Requirements

- Python 3.11+
- Install dependencies: `pip install -r requirements.txt`

## Project layout

- `src/common/` — shared utilities such as seeded RNG, elapsed+CPU timing, metrics, CSV I/O, and plotting
- `src/ackley/` — continuous Ackley warm-up benchmark
- `src/cvrp/` — CVRP model, distance matrix, construction heuristics, local search, and validation
- `src/cvrp/solvers/` — solver implementations for TS, ACO, SA, GA-Island, ALNS, and B&B/LDS
- `src/rushhour/` — Rush Hour board, A*, and baseline heuristics
- `src/gp/` — Genetic Programming using expression trees
- `src/gep/` — Gene Expression Programming using Karva genome to expression tree decoding
- `scripts/` — CLI entry points and experiment runners
- `configs/` — JSON parameter files
- `examples/` — tiny inputs for smoke runs
- `data/` — CVRPLIB benchmark instances added later
- `results/` — generated CSV outputs, ignored by Git except `.gitkeep`
- `plots/` — generated figures, ignored by Git except `.gitkeep`
- `tests/` — pytest tests

## CVRP benchmark instances

Official set — 6 instances only:

P-n16-k8, E-n22-k4, A-n32-k5, A-n80-k10, X-n101-k25, M-n200-k17.

## Stage 1-A: CVRP core

Basic CVRP parser, data model, distance matrix, cost calculation and feasibility
validation were added. A tiny sanity instance (known cost about 80.64) lives under
`examples/`.

Validate a solution file against an instance:

```
python scripts/validate_cvrp_solution.py --instance examples/tiny_cvrp.vrp --solution examples/tiny_solution.txt
```

Also print `0 0` lines for unused vehicles:

```
python scripts/validate_cvrp_solution.py --instance examples/tiny_cvrp.vrp --solution examples/tiny_solution.txt --include-unused-vehicles
```

## Stage 1-B: Rush Hour core

Basic Rush Hour board parsing (36-character strings), vehicle detection, legal move
generation, board features, baseline heuristics and a simple A* search with node and
time caps were added. A tiny one-move puzzle lives under `examples/`.

Solve a puzzle:

```
python scripts/solve_rushhour.py --puzzle examples/rushhour_tiny.txt
```

Choose a heuristic (`zero`, `blocking`, `blocking_distance`):

```
python scripts/solve_rushhour.py --puzzle examples/rushhour_tiny.txt --heuristic blocking_distance
```

## Stage 2-A: CVRP baseline

Clarke-Wright savings construction and simple local search were added. The
multi-stage baseline heuristic is:

construction (Clarke-Wright) -> 2-opt inside routes -> relocate between routes -> validation

Run it on the tiny example:

```
python scripts/run_cvrp_baseline.py --instance examples/tiny_cvrp.vrp
```

Optionally write the solution to a file:

```
python scripts/run_cvrp_baseline.py --instance examples/tiny_cvrp.vrp --output results/tiny_baseline_solution.txt
```

## Stage 2-B: Rush Hour evaluator

A small evaluator was added to test Rush Hour heuristics safely. Every A* call
has a node cap and a time cap, and a whole evaluation has a total time budget,
so a bad heuristic cannot hang the run (this protects the later GP/GEP stages).

```
python scripts/evaluate_rushhour_heuristics.py --puzzles examples/rushhour_eval_puzzles.txt
```

Optional CSV summary:

```
python scripts/evaluate_rushhour_heuristics.py --puzzles examples/rushhour_eval_puzzles.txt --output results/rushhour_heuristics.csv
```

## Stage 3-A: Ackley warm-up

The Ackley function and a small random-search sanity run were added. The
assignment default is d=10 with bounds [-32.768, 32.768]; the optimum is
f(0, ..., 0) = 0. Random search is only a sanity check for the objective,
bounds and seeding — it is not one of the six required algorithms.

```
python scripts/run_ackley_random_search.py --iterations 1000 --seed 42
```

Optional CSV output:

```
python scripts/run_ackley_random_search.py --iterations 1000 --seed 42 --output results/ackley_random_search.csv
```

## Usage

CLI entry points will be documented here as they are implemented in later stages.

All inputs are passed as command-line arguments. No hardcoded local paths.

## Reproducibility

Every stochastic run accepts a `--seed`.

For fair comparison, all algorithms receive the same timeout for the same CVRP instance.
