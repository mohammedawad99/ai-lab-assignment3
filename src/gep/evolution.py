"""GEP evolution loop for Rush Hour heuristics.

Every genome is decoded to a heuristic callable and scored by the safe Rush
Hour evaluator from Stage 2-B, so a bad genome can time out or fail but can
never crash or hang the run.
"""

import time
from dataclasses import dataclass

import numpy as np

from src.gep.decoder import decode_genome_to_tree
from src.gep.generator import make_initial_population
from src.gep.genome import GEPGenome
from src.gep.operators import (
    one_point_crossover,
    point_mutation,
    tournament_select,
    two_point_crossover,
)
from src.rushhour.evaluator import evaluate_heuristic, fitness_from_evaluation


@dataclass
class GEPIndividualResult:
    genome: GEPGenome
    fitness: float
    solved_count: int
    puzzle_count: int
    total_expanded_nodes: int
    total_cost: int
    expression: str
    genome_string: str


@dataclass
class GEPRunResult:
    algorithm: str
    seed: int
    generations: int
    population_size: int
    head_length: int
    best: GEPIndividualResult
    elapsed_time: float
    cpu_time: float
    history: list[dict]


def evaluate_genome(genome: GEPGenome, puzzles, max_nodes_per_puzzle: int = 10000,
                    max_time_per_puzzle_sec: float = 1.0,
                    max_total_time_sec: float = 10.0) -> GEPIndividualResult:
    """Decode and score one genome with the safe evaluator (per-genome budget)."""
    tree = decode_genome_to_tree(genome)
    expression = tree.to_string()
    evaluation = evaluate_heuristic(
        expression, tree.to_heuristic(), puzzles,
        max_nodes_per_puzzle=max_nodes_per_puzzle,
        max_time_per_puzzle_sec=max_time_per_puzzle_sec,
        max_total_time_sec=max_total_time_sec,
    )
    return GEPIndividualResult(
        genome=genome,
        fitness=fitness_from_evaluation(evaluation),
        solved_count=evaluation.solved_count,
        puzzle_count=evaluation.puzzle_count,
        total_expanded_nodes=evaluation.total_expanded_nodes,
        total_cost=evaluation.total_cost,
        expression=expression,
        genome_string=genome.to_string(),
    )


def run_gep(puzzles, generations: int = 10, population_size: int = 20,
            seed: int = 42, head_length: int = 6, crossover_rate: float = 0.8,
            mutation_rate: float = 0.05, max_nodes_per_puzzle: int = 10000,
            max_time_per_puzzle_sec: float = 1.0,
            max_total_time_sec: float = 10.0) -> GEPRunResult:
    start_elapsed = time.perf_counter()
    start_cpu = time.process_time()

    rng = np.random.default_rng(seed)
    caps = dict(max_nodes_per_puzzle=max_nodes_per_puzzle,
                max_time_per_puzzle_sec=max_time_per_puzzle_sec,
                max_total_time_sec=max_total_time_sec)

    population = make_initial_population(rng, population_size, head_length)
    results = [evaluate_genome(genome, puzzles, **caps) for genome in population]
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
                if rng.random() < 0.5:
                    child = one_point_crossover(child, other, rng)
                else:
                    child = two_point_crossover(child, other, rng)
            child = point_mutation(child, rng, mutation_rate)
            new_population.append(child)

        population = new_population
        results = [evaluate_genome(genome, puzzles, **caps) for genome in population]
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
            "best_genome": best.genome_string,
        })

    return GEPRunResult(
        algorithm="gep_rushhour",
        seed=seed,
        generations=completed,
        population_size=population_size,
        head_length=head_length,
        best=best,
        elapsed_time=time.perf_counter() - start_elapsed,
        cpu_time=time.process_time() - start_cpu,
        history=history,
    )
