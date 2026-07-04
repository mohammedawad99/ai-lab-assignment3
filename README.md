# AI Lab — Assignment 3: CVRP + Generative Heuristics (GP/GEP)

Course 203.3630 — Artificial Intelligence Lab.

Two parts:
- **Part A — CVRP:** solve the Capacitated Vehicle Routing Problem and compare six
  search algorithms (TS, ACO, SA, GA with Island Model, ALNS, Branch & Bound/LDS),
  plus at least one multi-stage heuristic. An Ackley-function warm-up validates each solver.
- **Part B — Generative AI:** use GP and GEP to evolve improved Rush Hour heuristics,
  evaluated by running A* with the candidate heuristic.

## Status

Stage 0 — scaffold only. No algorithmic code yet.

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

## Usage

CLI entry points will be documented here as they are implemented in later stages.

All inputs are passed as command-line arguments. No hardcoded local paths.

## Reproducibility

Every stochastic run accepts a `--seed`.

For fair comparison, all algorithms receive the same timeout for the same CVRP instance.
