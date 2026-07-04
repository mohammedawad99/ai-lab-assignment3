"""GP evolution loop for Rush Hour heuristics.

Every candidate tree is turned into a heuristic callable and scored by the
safe Rush Hour evaluator from Stage 2-B, so a bad tree can time out or fail
but can never crash or hang the run.
"""

import time
from dataclasses import dataclass

import numpy as np

from src.gp.generator import make_initial_population
from src.gp.operators import subtree_crossover, subtree_mutation, tournament_select
from src.gp.tree import GPTree
from src.rushhour.evaluator import evaluate_heuristic, fitness_from_evaluation


@dataclass
class GPIndividualResult:
    tree: GPTree
    fitness: float
    solved_count: int
    puzzle_count: int
    total_expanded_nodes: int
    total_cost: int
    expression: str


@dataclass
class GPRunResult:
    algorithm: str
    seed: int
    generations: int
    population_size: int
    best: GPIndividualResult
    elapsed_time: float
    cpu_time: float
    history: list[dict]


def evaluate_tree(tree: GPTree, puzzles, max_nodes_per_puzzle: int = 10000,
                  max_time_per_puzzle_sec: float = 1.0,
                  max_total_time_sec: float = 10.0) -> GPIndividualResult:
    """Score one tree with the safe evaluator (per-individual budget)."""
    expression = tree.to_string()
    evaluation = evaluate_heuristic(
        expression, tree.to_heuristic(), puzzles,
        max_nodes_per_puzzle=max_nodes_per_puzzle,
        max_time_per_puzzle_sec=max_time_per_puzzle_sec,
        max_total_time_sec=max_total_time_sec,
    )
    return GPIndividualResult(
        tree=tree,
        fitness=fitness_from_evaluation(evaluation),
        solved_count=evaluation.solved_count,
        puzzle_count=evaluation.puzzle_count,
        total_expanded_nodes=evaluation.total_expanded_nodes,
        total_cost=evaluation.total_cost,
        expression=expression,
    )


def run_gp(puzzles, generations: int = 10, population_size: int = 20,
           seed: int = 42, max_depth: int = 4, crossover_rate: float = 0.8,
           mutation_rate: float = 0.2, max_nodes_per_puzzle: int = 10000,
           max_time_per_puzzle_sec: float = 1.0,
           max_total_time_sec: float = 10.0) -> GPRunResult:
    start_elapsed = time.perf_counter()
    start_cpu = time.process_time()

    rng = np.random.default_rng(seed)
    caps = dict(max_nodes_per_puzzle=max_nodes_per_puzzle,
                max_time_per_puzzle_sec=max_time_per_puzzle_sec,
                max_total_time_sec=max_total_time_sec)
    # trees deeper than this are rejected by crossover/mutation
    operator_max_depth = max_depth + 2

    population = make_initial_population(rng, population_size, max_depth)
    results = [evaluate_tree(tree, puzzles, **caps) for tree in population]
    best = max(results, key=lambda r: r.fitness)

    history = []
    completed = 0
    for generation in range(1, generations + 1):
        fitnesses = [r.fitness for r in results]
        elite_index = max(range(len(results)), key=lambda i: fitnesses[i])
        new_population = [population[elite_index].copy()]
        while len(new_population) < population_size:
            child = tournament_select(population, fitnesses, rng)
            if rng.random() < crossover_rate:
                other = tournament_select(population, fitnesses, rng)
                child = subtree_crossover(child, other, rng, operator_max_depth)
            if rng.random() < mutation_rate:
                child = subtree_mutation(child, rng, operator_max_depth)
            new_population.append(child)

        population = new_population
        results = [evaluate_tree(tree, puzzles, **caps) for tree in population]
        generation_best = max(results, key=lambda r: r.fitness)
        if generation_best.fitness > best.fitness:
            best = generation_best
        completed = generation

        history.append({
            "generation": generation,
            "best_fitness": best.fitness,
            "best_solved_count": best.solved_count,
            "best_total_expanded_nodes": best.total_expanded_nodes,
            "best_total_cost": best.total_cost,
            "best_expression": best.expression,
        })

    return GPRunResult(
        algorithm="gp_rushhour",
        seed=seed,
        generations=completed,
        population_size=population_size,
        best=best,
        elapsed_time=time.perf_counter() - start_elapsed,
        cpu_time=time.process_time() - start_cpu,
        history=history,
    )
