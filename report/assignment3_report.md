# Assignment 3 Report

Course 203.3630 — Artificial Intelligence Lab.

This file is the report skeleton. Sections marked [fill after final run]
are completed only after the final experiments have been executed.

## 1. Introduction

- What the assignment asks: [fill after final run — short problem statement]
- Part A: solve the Ackley warm-up function and the CVRP with six search
  algorithms, plus a multi-stage heuristic.
- Part B: evolve Rush Hour heuristics with GP and GEP and evaluate them by
  running A*.

## 2. Implementation Overview

### 2.1 Project structure

- `src/common/`, `src/ackley/`, `src/cvrp/` (+ `src/cvrp/solvers/`),
  `src/rushhour/`, `src/gp/`, `src/gep/`, `src/experiments/`
- `scripts/` for CLI entry points, `tests/` for pytest, `configs/` for JSON
  settings, `data/` for benchmark files, `results/` for generated output.
- [fill after final run — one short paragraph on how the parts connect]

### 2.2 Reproducibility and command-line interface

- Every stochastic run takes a `--seed`; all inputs come from CLI arguments.
- Timeouts are CLI arguments so all algorithms get the same time budget on
  the same instance.
- [fill after final run — exact command list reference]

### 2.3 Validation and safety checks

- CVRP solutions are validated for depot structure, missing/duplicate
  customers, capacity, and vehicle count.
- Rush Hour heuristic evaluation has node caps, per-puzzle time caps, and a
  total time budget, so a bad heuristic cannot hang a run.
- [fill after final run — anything that fired during the final runs]

## 3. Part A — Ackley Function

The Ackley function for d dimensions:

f(x) = -a * exp(-b * sqrt((1/d) * sum(x_i^2)))
       - exp((1/d) * sum(cos(c * x_i))) + a + e

with a = 20, b = 0.2, c = 2*pi.

- Dimension: d = 10.
- Bounds: x_i in [-32.768, 32.768] for all i.
- Known optimum: f(0, ..., 0) = 0.

Results table: [fill after final run]

Short analysis: [fill after final run]

## 4. Part A — CVRP

Official instances and best-known solutions (BKS):

| instance | BKS cost |
| --- | --- |
| P-n16-k8 | 450 |
| E-n22-k4 | 375 |
| A-n32-k5 | 784 |
| A-n80-k10 | 1763 |
| X-n101-k25 | 27591 |
| M-n200-k17 | 1275 |

The BKS values are used only to compute the gap percentage of our results;
they are not used by the algorithms themselves.

Result table: [fill after final run]

Route feasibility discussion: [fill after final run — include the
vehicle-count repair story from P-n16-k8]

## 5. Multi-stage CVRP Heuristic

The baseline heuristic runs in stages:

1. Construction with Clarke-Wright savings.
2. 2-opt improvement inside each route.
3. Relocate improvement between routes.
4. Vehicle-count feasibility repair (empty surplus routes and reinsert their
   customers; rebuild by cheapest insertion as a fallback).
5. Final validation.

Complexity discussion: [fill after final run — rough cost of each stage]

## 6. CVRP Algorithms

### 6.1 Simulated Annealing

- Representation: full solutions; neighbors from random relocate/swap/2-opt.
- Main parameters: initial temperature, cooling rate, iterations, timeout.
- Behavior/results: [fill after final run]

### 6.2 Tabu Search

- Representation: full solutions; candidate sampling from the same
  neighborhood; tabu key is the customer sequence per route; aspiration on
  a new global best.
- Main parameters: tabu tenure, candidates per iteration, iterations.
- Behavior/results: [fill after final run]

### 6.3 Ant Colony Optimization

- Representation: ants build routes customer by customer using pheromone
  and inverse distance; capacity-aware; light 2-opt on ant solutions.
- Main parameters: ants, alpha, beta, evaporation rate, deposit.
- Behavior/results: [fill after final run]

### 6.4 GA Island Model

- Representation: giant-tour chromosome with a capacity-aware split;
  OX crossover; swap/inversion mutation; ring migration between islands.
- Main parameters: population size, islands, migration interval, rates.
- Behavior/results: [fill after final run]

### 6.5 ALNS

- Representation: destroy operators (random/worst removal) and repair
  operators (greedy/regret-2 insertion) with adaptive weights and
  simulated-annealing acceptance.
- Main parameters: removal fraction, reaction rate, temperature schedule.
- Behavior/results: [fill after final run]

### 6.6 Branch-and-Bound / LDS

- Representation: time-limited search over customer insertion decisions,
  hardest customers first; discrepancy limit; partial-cost bound against
  the incumbent from the baseline.
- Main parameters: max discrepancy, max nodes, timeout.
- Behavior/results: [fill after final run]

## 7. Ackley Adaptations

- SA and Tabu Search are natural continuous adaptations (Gaussian steps,
  rounded-point tabu list).
- ACO and B&B/LDS are discretized adaptations (bins per dimension); they
  are honest warm-up variants, not canonical continuous versions.
- GA-Island and ALNS work directly on continuous candidate vectors (blend
  crossover / dimension-wise destroy and repair).
- Note: [fill after final run — which adaptations behaved reasonably]

## 8. Part B — Rush Hour with GP and GEP

- A* solver with g = number of moves and pluggable heuristics.
- Heuristic candidates are evaluated by running A* on a puzzle set through
  a safe evaluator (node cap, per-puzzle time cap, total time budget,
  exceptions recorded as heuristic errors).
- Fitness summary: solved count strongly rewarded; expanded nodes, solution
  cost, timeouts, and node-cap hits penalized.
- GP representation: expression trees over board features (distance to
  exit, blocking cars, free exit cells) and small constants, protected math.
- GEP representation: linear head/tail genome decoded as a Karva
  K-expression into the same kind of expression.
- Comparison metrics: solution quality, diversity of best
  expressions/genomes, creation (evolution) time, solved puzzles, expanded
  nodes, total solution cost.

## 9. Results

### 9.1 Ackley results

[fill after final run — table + short comments]

### 9.2 CVRP results

[fill after final run — per-instance table with best/mean cost and gap]

### 9.3 Rush Hour GP/GEP results

[fill after final run — comparison table, best expressions/genomes]

### 9.4 Generated tables and plots

[fill after final run — reference the generated assets under
results/final_experiments/report_assets]

## 10. Analysis and Discussion

- Which algorithms were strong on small vs. large CVRP instances:
  [fill after final run]
- How gaps change with instance size: [fill after final run]
- What happened on Ackley: [fill after final run]
- GP vs GEP comparison: [fill after final run]
- Runtime vs quality tradeoff: [fill after final run]
- Limitations: [fill after final run]

## 11. Complexity and Practical Considerations

- CVRP feasibility checking cost: [fill after final run]
- Local search neighborhood costs: [fill after final run]
- Variance across stochastic runs and seeds: [fill after final run]
- Timeouts and fair comparison between algorithms: [fill after final run]

## 12. Use of AI Tools

AI tools were used as coding and debugging assistants during the project.
The final code and this report were reviewed and understood by the student.
Any generated code was tested and adjusted through the project's test suite
and staged workflow.

## 13. Reproducibility

- Python version: [fill after final run]
- Install dependencies: `pip install -r requirements.txt`
- Smoke check: `python scripts/run_smoke_suite.py --output-dir results/smoke_suite`
- Final experiment commands: [fill after final run — path to the generated
  command file, e.g. results/final_experiment_commands.txt]
- Official CVRP data: place the six official `.vrp` files under
  `data/official_cvrp/` and verify with
  `python scripts/check_official_cvrp_data.py --strict`.

## 14. Conclusion

[fill after final run]
