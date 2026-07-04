# AI Lab — Assignment 3: CVRP + Generative Heuristics (GP/GEP)

Course 203.3630 — Artificial Intelligence Lab.

Two parts:
- **Part A — CVRP:** solve the Capacitated Vehicle Routing Problem and compare six
  search algorithms (TS, ACO, SA, GA with Island Model, ALNS, Branch & Bound/LDS),
  plus at least one multi-stage heuristic. An Ackley-function warm-up validates each solver.
- **Part B — Generative AI:** use GP and GEP to evolve improved Rush Hour heuristics,
  evaluated by running A* with the candidate heuristic.

## Status

Stage 5-A — ALNS foundations done (CVRP + Ackley). No B&B/LDS or GP/GEP yet.

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

## Stage 3-B: SA and Tabu foundations

Simulated Annealing and Tabu Search were added as the first two working
metaheuristic foundations (not tuned yet). The CVRP versions start from the
multi-stage baseline solution and share a random relocate/swap/2-opt
neighborhood. The Ackley versions reuse the Ackley framework and bounds.

CVRP:

```
python scripts/run_cvrp_sa_ts.py --instance examples/tiny_cvrp.vrp --algorithm sa
python scripts/run_cvrp_sa_ts.py --instance examples/tiny_cvrp.vrp --algorithm tabu
```

Ackley:

```
python scripts/run_ackley_sa_ts.py --algorithm both --iterations 1000 --seed 42
```

## Stage 4-A: ACO foundations

Ant Colony Optimization was added for CVRP: ants build capacity-feasible
solutions with probabilities based on pheromone and inverse distance, with
evaporation and deposits on the iteration-best and global-best solutions.
Because Ackley is continuous, the Ackley version is only a simple
discretized ACO-style variant (bins per dimension), used for the warm-up.

CVRP:

```
python scripts/run_cvrp_aco.py --instance examples/tiny_cvrp.vrp --iterations 100 --ants 20 --seed 42
```

Ackley:

```
python scripts/run_ackley_aco.py --iterations 100 --ants 20 --seed 42
```

## Stage 4-B: GA Island Model foundations

A Genetic Algorithm with an Island Model was added for CVRP and Ackley.
The CVRP version uses a giant-tour chromosome (a permutation of all
customers) with a capacity-aware split into routes; Ackley uses continuous
vectors with blend crossover and Gaussian mutation. Islands evolve
separately and exchange their best individuals in a ring every few
generations.

CVRP:

```
python scripts/run_cvrp_ga_island.py --instance examples/tiny_cvrp.vrp --generations 100 --population-size 30 --islands 4 --seed 42
```

Ackley:

```
python scripts/run_ackley_ga_island.py --generations 100 --population-size 30 --islands 4 --seed 42
```

## Stage 5-A: ALNS foundations

Adaptive Large Neighborhood Search was added for CVRP: destroy operators
(random removal, worst removal) and repair operators (greedy insertion,
regret-2 insertion) are picked by adaptive weights, with simulated-annealing
acceptance. A small ALNS-style version was also added for Ackley by
destroying and repairing vector dimensions — a practical warm-up adaptation,
since ALNS is really meant for combinatorial problems.

CVRP:

```
python scripts/run_cvrp_alns.py --instance examples/tiny_cvrp.vrp --iterations 100 --seed 42
```

Ackley:

```
python scripts/run_ackley_alns.py --iterations 100 --seed 42
```

## Usage

CLI entry points will be documented here as they are implemented in later stages.

All inputs are passed as command-line arguments. No hardcoded local paths.

## Reproducibility

Every stochastic run accepts a `--seed`.

For fair comparison, all algorithms receive the same timeout for the same CVRP instance.
