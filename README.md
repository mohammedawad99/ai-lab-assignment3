# AI Lab — Assignment 3: CVRP + Generative Heuristics (GP/GEP)

Course 203.3630 — Artificial Intelligence Lab.

Two parts:
- **Part A — CVRP:** solve the Capacitated Vehicle Routing Problem and compare six
  search algorithms (TS, ACO, SA, GA with Island Model, ALNS, Branch & Bound/LDS),
  plus at least one multi-stage heuristic. An Ackley-function warm-up validates each solver.
- **Part B — Generative AI:** use GP and GEP to evolve improved Rush Hour heuristics,
  evaluated by running A* with the candidate heuristic.

## Status

Stage 9-C2 — visual final report (Markdown + PDF with figures) under `report/`, submission audit in place.

Report figures live under `report/figures/` and the visual PDF can be
regenerated with:

```
python scripts/generate_report_figures.py
python scripts/export_report_pdf.py
```

The final audit can be run with:

```
python scripts/audit_submission.py --check-results --check-pdf
```

Generated experiment outputs live under `results/final_experiments/` and are
not committed automatically. The official `.vrp` files are user-provided
data and are not committed unless the course explicitly asks for them.

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

construction (Clarke-Wright) -> 2-opt inside routes -> relocate between routes -> vehicle-count repair -> validation

The repair stage empties surplus routes and reinserts their customers,
because Clarke-Wright can need more routes than the instance allows
(this happened on the official P-n16-k8).

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

## Stage 5-B: B&B / LDS foundations

A time-limited Branch and Bound / Limited Discrepancy Search was added for
CVRP. It searches over customer insertion decisions (hardest customers
first): the cheapest insertion costs 0 discrepancy, the next 1, and so on,
with a partial-cost bound against the incumbent. It starts from the
multi-stage baseline as the incumbent and always returns the best feasible
incumbent found. It is exact-inspired and time-limited, not a full exact
CVRP solver for the large instances. A small discretized LDS-style version
was also added for Ackley (bins per dimension, ranked by distance to zero).

CVRP:

```
python scripts/run_cvrp_bnb_lds.py --instance examples/tiny_cvrp.vrp --max-discrepancy 3 --max-nodes 50000 --seed 42
```

Ackley:

```
python scripts/run_ackley_bnb_lds.py --dimension 10 --bins-per-dimension 7 --max-discrepancy 3 --seed 42
```

## Stage 6-A: GP for Rush Hour heuristics

A basic Genetic Programming system was added to evolve Rush Hour heuristic
expressions. Terminals come from the simple board features (distance to
exit, blocking cars, free exit cells) plus small constants; operators are
protected (+, -, *, /, min, max, abs, neg, log). Every candidate is turned
into a heuristic function and scored by running A* through the safe
evaluator, with node and time caps, so a bad heuristic cannot hang the run.

```
python scripts/run_gp_rushhour.py --puzzles examples/rushhour_gp_train.txt --generations 10 --population-size 20 --seed 42
```

Optional history and best-expression output:

```
python scripts/run_gp_rushhour.py --puzzles examples/rushhour_gp_train.txt --output results/gp_rushhour_history.csv --best-output results/gp_best_expression.txt
```

## Stage 6-B: GEP for Rush Hour heuristics

A basic Gene Expression Programming system was added. Unlike GP, GEP uses a
linear genome with a head (functions or terminals) and a tail (terminals
only, length head+1), decoded as a Karva/K-expression into a heuristic
expression. Candidates are evaluated exactly like the GP ones: A* through
the safe evaluator with node and time caps, so bad heuristics cannot hang
the run. The training set is the same 4 puzzles as GP for fair comparison.

```
python scripts/run_gep_rushhour.py --puzzles examples/rushhour_gep_train.txt --generations 10 --population-size 20 --head-length 6 --seed 42
```

Optional history and best-expression output:

```
python scripts/run_gep_rushhour.py --puzzles examples/rushhour_gep_train.txt --output results/gep_rushhour_history.csv --best-output results/gep_best_expression.txt
```

## Stage 6-C: GP vs GEP comparison

A comparison runner was added for the GP and GEP Rush Hour heuristics. Both
methods run with the same seeds, the same train/evaluation puzzle sets, and
the same evaluation caps. The comparison records eval fitness, solved
counts, expanded nodes, total solution cost, best expression/genome,
expression/genome diversity across seeds, and runtime. Results describe the
supplied puzzle sets and settings only.

```
python scripts/compare_gp_gep_rushhour.py --train-puzzles examples/rushhour_gp_train.txt --eval-puzzles examples/rushhour_gp_gep_eval.txt --seeds 42 43 44
```

Optional CSV/summary output:

```
python scripts/compare_gp_gep_rushhour.py --train-puzzles examples/rushhour_gp_train.txt --eval-puzzles examples/rushhour_gp_gep_eval.txt --output results/gp_gep_comparison_runs.csv --summary-output results/gp_gep_comparison_summary.txt
```

## Stage 7-A: Unified experiment runner

A unified runner was added for CVRP and Ackley experiments. It runs the six
required algorithms (sa, tabu, aco, ga_island, alns, bnb_lds) with shared
seeds, budget, and timeout, and writes one CSV row per run. The multi-stage
baseline (`--include-baseline`) and random search (`--include-random-search`)
are optional reference methods, not part of the six required algorithms.

CVRP smoke run:

```
python scripts/run_experiments.py --part cvrp --instances examples/tiny_cvrp.vrp --algorithms sa tabu aco --seeds 42 --budget 20 --timeout 5 --output results/cvrp_smoke.csv
```

Ackley smoke run:

```
python scripts/run_experiments.py --part ackley --algorithms sa tabu aco --seeds 42 --budget 20 --timeout 5 --dimension 4 --output results/ackley_smoke.csv
```

Both parts:

```
python scripts/run_experiments.py --part both --instances examples/tiny_cvrp.vrp --algorithms sa tabu aco --seeds 42 --budget 20 --timeout 5 --dimension 4 --output-dir results/smoke_experiments
```

## Stage 7-B: Official CVRP benchmark setup

The project now has a BKS (best-known-solution) table for the six official
CVRP instances under `data/cvrp_bks.csv`. The official `.vrp` files are not
part of the repository — place them manually under:

```
data/official_cvrp/
```

Expected filenames:

- P-n16-k8.vrp
- E-n22-k4.vrp
- A-n32-k5.vrp
- A-n80-k10.vrp
- X-n101-k25.vrp
- M-n200-k17.vrp

Only the six listed instances are expected. A checker verifies the setup and
can be run before the data is placed:

```
python scripts/check_official_cvrp_data.py
```

Strict mode requires all six files to be present and parseable:

```
python scripts/check_official_cvrp_data.py --strict
```

Optional CSV report:

```
python scripts/check_official_cvrp_data.py --csv-output results/official_cvrp_data_check.csv
```

## Stage 8-A: Smoke experiments and summaries

A smoke-suite runner was added to check all implemented CVRP and Ackley
algorithms end-to-end with small budgets — it is a fast sanity run, not the
final benchmark. Raw result CSVs and compact summary CSVs (grouped per
instance/algorithm) are written separately.

```
python scripts/run_smoke_suite.py --output-dir results/smoke_suite
```

Summarize a raw results CSV on its own:

```
python scripts/summarize_experiments.py --part cvrp --input results/smoke_suite/raw/cvrp_results.csv --output results/smoke_suite/summary/cvrp_summary.csv
```

When official instances are used, the BKS table can be passed to the unified
runner to fill the gap column:

```
python scripts/run_experiments.py --part cvrp --instances data/official_cvrp/P-n16-k8.vrp --algorithms sa tabu --seeds 42 --budget 100 --timeout 30 --bks data/cvrp_bks.csv --output results/example_with_bks.csv
```

## Stage 8-B: Official mini-run

A mini-run script was added for the official CVRP data. It first checks
which official files are actually present under `data/official_cvrp/`. If no
files are placed yet, it can skip cleanly:

```
python scripts/run_official_mini_experiment.py --allow-missing-data
```

Once the files are placed, run the readiness mini-run:

```
python scripts/run_official_mini_experiment.py --instances P-n16-k8 --algorithms baseline sa tabu aco ga_island alns bnb_lds --seeds 42 --budget 20 --timeout 30 --output-dir results/official_mini
```

This mini-run is only a readiness check, not the final benchmark.

## Stage 8-C: Report assets

Report-asset utilities were added for tables and plots. They read existing
experiment CSV files and write markdown tables and PNG charts (or plain-text
note files when data is missing). The generated assets go under `results/`
and are not the final report.

```
python scripts/generate_report_assets.py --cvrp-raw results/smoke_suite/raw/cvrp_results.csv --cvrp-summary results/smoke_suite/summary/cvrp_summary.csv --ackley-raw results/smoke_suite/raw/ackley_results.csv --ackley-summary results/smoke_suite/summary/ackley_summary.csv --output-dir results/report_assets
```

The same command can point at official mini-run CSVs, for example
`results/official_mini_after_repair/raw/official_mini_cvrp_results.csv` and
`results/official_mini_after_repair/summary/official_mini_cvrp_summary.csv`.

## Stage 9-A: Final plan and report skeleton

A final experiment plan lives in `configs/final_experiment_plan.json`
(per-instance budgets and timeouts, shared seeds). Check and print it with:

```
python scripts/print_final_experiment_plan.py
```

Write the exact run commands to a file:

```
python scripts/print_final_experiment_plan.py --write-commands results/final_experiment_commands.txt
```

If the official CVRP data is placed, stricter checking also verifies the files:

```
python scripts/print_final_experiment_plan.py --require-official-data
```

`report/assignment3_report.md` is only a skeleton — it is filled after the
final results exist.

## Stage 9-B: Final experiment execution

A resumable final experiment runner executes the plan from
`configs/final_experiment_plan.json`. It writes raw CSVs, summary CSVs,
report assets, and an execution manifest under `results/final_experiments/`.

```
python scripts/run_final_experiments.py
```

To resume safely after an interruption, run the same command again —
existing raw CSVs with rows are skipped. Generated results are not
committed automatically.

## Usage

CLI entry points will be documented here as they are implemented in later stages.

All inputs are passed as command-line arguments. No hardcoded local paths.

## Reproducibility

Every stochastic run accepts a `--seed`.

For fair comparison, all algorithms receive the same timeout for the same CVRP instance.
