"""Comparison runner for GP vs GEP Rush Hour heuristics.

Runs both methods with the same seeds, puzzle sets and evaluation caps, then
summarizes quality, solved counts, expanded nodes, runtime and diversity.
The results only describe the supplied puzzle sets and settings — they are
not a claim that one method is better in general.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

from src.gep.evolution import evaluate_genome, run_gep
from src.gp.evolution import evaluate_tree, run_gp


@dataclass
class AlgorithmRunSummary:
    algorithm: str
    seed: int
    generations: int
    population_size: int
    train_fitness: float
    train_solved_count: int
    train_puzzle_count: int
    train_total_expanded_nodes: int
    train_total_cost: int
    eval_fitness: float
    eval_solved_count: int
    eval_puzzle_count: int
    eval_total_expanded_nodes: int
    eval_total_cost: int
    elapsed_time: float
    cpu_time: float
    best_expression: str
    best_genome: str | None


@dataclass
class ComparisonSummary:
    gp_runs: list[AlgorithmRunSummary]
    gep_runs: list[AlgorithmRunSummary]
    gp_expression_diversity: float
    gep_expression_diversity: float
    gep_genome_diversity: float
    best_gp: AlgorithmRunSummary
    best_gep: AlgorithmRunSummary


# ---------- diversity ----------

def expression_diversity(expressions: list[str]) -> float:
    """Unique best expressions divided by the number of runs."""
    if not expressions:
        return 0.0
    return len(set(expressions)) / len(expressions)


def genome_diversity(genomes: list[str | None]) -> float:
    """Unique best genomes divided by the number of runs (None entries ignored)."""
    values = [genome for genome in genomes if genome is not None]
    if not values:
        return 0.0
    return len(set(values)) / len(values)


# ---------- per-run summaries ----------

def summarize_gp_run(run_result, eval_puzzles, max_nodes_per_puzzle,
                     max_time_per_puzzle_sec, max_total_time_sec) -> AlgorithmRunSummary:
    evaluation = evaluate_tree(
        run_result.best.tree, eval_puzzles,
        max_nodes_per_puzzle=max_nodes_per_puzzle,
        max_time_per_puzzle_sec=max_time_per_puzzle_sec,
        max_total_time_sec=max_total_time_sec,
    )
    return AlgorithmRunSummary(
        algorithm=run_result.algorithm,
        seed=run_result.seed,
        generations=run_result.generations,
        population_size=run_result.population_size,
        train_fitness=run_result.best.fitness,
        train_solved_count=run_result.best.solved_count,
        train_puzzle_count=run_result.best.puzzle_count,
        train_total_expanded_nodes=run_result.best.total_expanded_nodes,
        train_total_cost=run_result.best.total_cost,
        eval_fitness=evaluation.fitness,
        eval_solved_count=evaluation.solved_count,
        eval_puzzle_count=evaluation.puzzle_count,
        eval_total_expanded_nodes=evaluation.total_expanded_nodes,
        eval_total_cost=evaluation.total_cost,
        elapsed_time=run_result.elapsed_time,
        cpu_time=run_result.cpu_time,
        best_expression=run_result.best.expression,
        best_genome=None,
    )


def summarize_gep_run(run_result, eval_puzzles, max_nodes_per_puzzle,
                      max_time_per_puzzle_sec, max_total_time_sec) -> AlgorithmRunSummary:
    evaluation = evaluate_genome(
        run_result.best.genome, eval_puzzles,
        max_nodes_per_puzzle=max_nodes_per_puzzle,
        max_time_per_puzzle_sec=max_time_per_puzzle_sec,
        max_total_time_sec=max_total_time_sec,
    )
    return AlgorithmRunSummary(
        algorithm=run_result.algorithm,
        seed=run_result.seed,
        generations=run_result.generations,
        population_size=run_result.population_size,
        train_fitness=run_result.best.fitness,
        train_solved_count=run_result.best.solved_count,
        train_puzzle_count=run_result.best.puzzle_count,
        train_total_expanded_nodes=run_result.best.total_expanded_nodes,
        train_total_cost=run_result.best.total_cost,
        eval_fitness=evaluation.fitness,
        eval_solved_count=evaluation.solved_count,
        eval_puzzle_count=evaluation.puzzle_count,
        eval_total_expanded_nodes=evaluation.total_expanded_nodes,
        eval_total_cost=evaluation.total_cost,
        elapsed_time=run_result.elapsed_time,
        cpu_time=run_result.cpu_time,
        best_expression=run_result.best.expression,
        best_genome=run_result.best.genome_string,
    )


def _best_run(runs: list[AlgorithmRunSummary]) -> AlgorithmRunSummary:
    """Highest eval fitness, then more solved, then fewer expanded nodes."""
    return max(runs, key=lambda r: (r.eval_fitness, r.eval_solved_count,
                                    -r.eval_total_expanded_nodes))


# ---------- comparison ----------

def compare_gp_gep(train_puzzles, eval_puzzles, seeds, generations=10,
                   population_size=20, gp_max_depth=4, gep_head_length=6,
                   crossover_rate=0.8, gp_mutation_rate=0.2,
                   gep_mutation_rate=0.05, max_nodes_per_puzzle=10000,
                   max_time_per_puzzle_sec=1.0,
                   max_total_time_sec=10.0) -> ComparisonSummary:
    caps = dict(max_nodes_per_puzzle=max_nodes_per_puzzle,
                max_time_per_puzzle_sec=max_time_per_puzzle_sec,
                max_total_time_sec=max_total_time_sec)

    gp_runs = []
    gep_runs = []
    for seed in seeds:
        gp_result = run_gp(
            train_puzzles, generations=generations, population_size=population_size,
            seed=seed, max_depth=gp_max_depth, crossover_rate=crossover_rate,
            mutation_rate=gp_mutation_rate, **caps,
        )
        gp_runs.append(summarize_gp_run(gp_result, eval_puzzles, **caps))

        gep_result = run_gep(
            train_puzzles, generations=generations, population_size=population_size,
            seed=seed, head_length=gep_head_length, crossover_rate=crossover_rate,
            mutation_rate=gep_mutation_rate, **caps,
        )
        gep_runs.append(summarize_gep_run(gep_result, eval_puzzles, **caps))

    return ComparisonSummary(
        gp_runs=gp_runs,
        gep_runs=gep_runs,
        gp_expression_diversity=expression_diversity([r.best_expression for r in gp_runs]),
        gep_expression_diversity=expression_diversity([r.best_expression for r in gep_runs]),
        gep_genome_diversity=genome_diversity([r.best_genome for r in gep_runs]),
        best_gp=_best_run(gp_runs),
        best_gep=_best_run(gep_runs),
    )


# ---------- output ----------

CSV_HEADER = [
    "algorithm", "seed", "generations", "population_size",
    "train_fitness", "train_solved_count", "train_puzzle_count",
    "train_total_expanded_nodes", "train_total_cost",
    "eval_fitness", "eval_solved_count", "eval_puzzle_count",
    "eval_total_expanded_nodes", "eval_total_cost",
    "elapsed_time", "cpu_time", "best_expression", "best_genome",
]


def write_run_summaries_csv(summary: ComparisonSummary, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        for run in summary.gp_runs + summary.gep_runs:
            writer.writerow([
                run.algorithm, run.seed, run.generations, run.population_size,
                run.train_fitness, run.train_solved_count, run.train_puzzle_count,
                run.train_total_expanded_nodes, run.train_total_cost,
                run.eval_fitness, run.eval_solved_count, run.eval_puzzle_count,
                run.eval_total_expanded_nodes, run.eval_total_cost,
                f"{run.elapsed_time:.6f}", f"{run.cpu_time:.6f}",
                run.best_expression, run.best_genome or "",
            ])


def write_comparison_summary_txt(summary: ComparisonSummary, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"gp runs: {len(summary.gp_runs)}",
        f"gep runs: {len(summary.gep_runs)}",
        f"gp expression diversity: {summary.gp_expression_diversity:.2f}",
        f"gep expression diversity: {summary.gep_expression_diversity:.2f}",
        f"gep genome diversity: {summary.gep_genome_diversity:.2f}",
        f"best gp expression: {summary.best_gp.best_expression}",
        f"best gep expression: {summary.best_gep.best_expression}",
        f"best gep genome: {summary.best_gep.best_genome}",
        "These results depend on the small puzzle set and the selected random seeds.",
    ]
    path.write_text("\n".join(lines) + "\n")
