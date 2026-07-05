"""Fitness and evolution loops for the direct (no-A*) GP/GEP planner.

A policy expression is a GP tree (or a decoded GEP genome) over the move
features from direct_features.py; it is rolled out greedily with
direct_planner.greedy_policy_rollout. Nothing in this module imports or
calls the A* solver — that is the whole point of the bonus, and
tests/test_rushhour_direct_planner.py enforces it.

Fitness (higher = better), kept explainable:
  per solved puzzle:   +10000, +10 per step saved under max_steps
  per unsolved puzzle: -100 * (red distance + blockers) at the final state
  per repeated state:  -5
  parsimony:           -2 * expression size (once per individual)
"""

import time
from dataclasses import dataclass, field

import numpy as np

from src.gep.decoder import decode_genome_to_tree
from src.gep.generator import make_initial_population as gep_population
from src.gep.operators import (
    one_point_crossover,
    point_mutation,
    two_point_crossover,
)
from src.gep.operators import tournament_select as gep_select
from src.gep.symbols import DIRECT_TERMINALS
from src.gp.generator import make_initial_population as gp_population
from src.gp.operators import subtree_crossover, subtree_mutation
from src.gp.operators import tournament_select as gp_select
from src.rushhour.direct_features import DIRECT_FEATURE_NAMES, move_features
from src.rushhour.direct_planner import DirectRolloutResult, greedy_policy_rollout
from src.rushhour.features import blocking_cars, red_car_distance_to_exit

SOLVED_BONUS = 10000.0
STEP_BONUS = 10.0
UNSOLVED_PENALTY = 100.0
REPEAT_PENALTY = 5.0
SIZE_PENALTY = 2.0


def policy_from_tree(tree):
    """Turn a GP tree over DIRECT_FEATURE_NAMES into a move-scoring policy."""
    def score(state, move, after, context):
        features = move_features(state, move, after, context["visited"])
        return tree.evaluate(features)
    return score


@dataclass
class DirectEvaluation:
    solved_count: int
    puzzle_count: int
    total_steps_solved: int
    total_repeated_states: int
    total_runtime: float
    results: list[DirectRolloutResult] = field(default_factory=list)


def evaluate_direct_policy(score_move, puzzles, max_steps: int = 120,
                           timeout_per_puzzle: float = 2.0,
                           max_total_time_sec: float | None = None) -> DirectEvaluation:
    """Roll the policy out on every puzzle (with per-puzzle and optional
    total time caps, mirroring the safe A* evaluator's budget style)."""
    start = time.perf_counter()
    results = []
    for puzzle in puzzles:
        if max_total_time_sec is not None and \
                time.perf_counter() - start > max_total_time_sec:
            results.append(DirectRolloutResult(
                False, 0, 0, False, 0.0, puzzle, stopped_reason="budget"))
            continue
        results.append(greedy_policy_rollout(puzzle, score_move,
                                             max_steps=max_steps,
                                             timeout_sec=timeout_per_puzzle))
    return DirectEvaluation(
        solved_count=sum(1 for r in results if r.solved),
        puzzle_count=len(results),
        total_steps_solved=sum(r.steps for r in results if r.solved),
        total_repeated_states=sum(r.repeated_states for r in results),
        total_runtime=time.perf_counter() - start,
        results=results,
    )


def direct_fitness_value(evaluation: DirectEvaluation, expression_size: int,
                         max_steps: int = 120) -> float:
    score = 0.0
    for r in evaluation.results:
        if r.solved:
            score += SOLVED_BONUS + STEP_BONUS * max(0, max_steps - r.steps)
        else:
            remaining = (red_car_distance_to_exit(r.final_state)
                         + blocking_cars(r.final_state))
            score -= UNSOLVED_PENALTY * remaining
        score -= REPEAT_PENALTY * r.repeated_states
    return score - SIZE_PENALTY * expression_size


@dataclass
class DirectIndividual:
    expression: str
    fitness: float
    solved_count: int
    puzzle_count: int
    total_steps_solved: int
    total_repeated_states: int
    expression_size: int
    genome_string: str | None = None
    tree: object = None


@dataclass
class DirectRunResult:
    algorithm: str  # "gp_direct" / "gep_direct"
    seed: int
    generations: int
    population_size: int
    best: DirectIndividual
    elapsed_time: float
    expressions: list[str] = field(default_factory=list)  # final population


def _evaluate_tree_direct(tree, puzzles, max_steps, timeout_per_puzzle,
                          budget, genome_string=None) -> DirectIndividual:
    evaluation = evaluate_direct_policy(policy_from_tree(tree), puzzles,
                                        max_steps=max_steps,
                                        timeout_per_puzzle=timeout_per_puzzle,
                                        max_total_time_sec=budget)
    size = tree.size()
    return DirectIndividual(
        expression=tree.to_string(),
        fitness=direct_fitness_value(evaluation, size, max_steps),
        solved_count=evaluation.solved_count,
        puzzle_count=evaluation.puzzle_count,
        total_steps_solved=evaluation.total_steps_solved,
        total_repeated_states=evaluation.total_repeated_states,
        expression_size=size,
        genome_string=genome_string,
        tree=tree,
    )


def run_gp_direct(puzzles, generations: int = 20, population_size: int = 30,
                  seed: int = 42, max_depth: int = 4, crossover_rate: float = 0.8,
                  mutation_rate: float = 0.2, max_steps: int = 120,
                  timeout_per_puzzle: float = 2.0,
                  budget_per_individual: float = 6.0) -> DirectRunResult:
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    operator_max_depth = max_depth + 2

    population = gp_population(rng, population_size, max_depth,
                               terminals=DIRECT_FEATURE_NAMES)

    def score_all(pop):
        return [_evaluate_tree_direct(t, puzzles, max_steps,
                                      timeout_per_puzzle, budget_per_individual)
                for t in pop]

    results = score_all(population)
    best = max(results, key=lambda r: r.fitness)
    for _generation in range(1, generations + 1):
        fitnesses = [r.fitness for r in results]
        elite = max(range(len(results)), key=lambda i: fitnesses[i])
        new_population = [population[elite].copy()]
        while len(new_population) < population_size:
            child = gp_select(population, fitnesses, rng)
            if rng.random() < crossover_rate:
                other = gp_select(population, fitnesses, rng)
                child = subtree_crossover(child, other, rng, operator_max_depth)
            if rng.random() < mutation_rate:
                child = subtree_mutation(child, rng, operator_max_depth,
                                         terminals=DIRECT_FEATURE_NAMES)
            new_population.append(child)
        population = new_population
        results = score_all(population)
        generation_best = max(results, key=lambda r: r.fitness)
        if generation_best.fitness > best.fitness:
            best = generation_best

    return DirectRunResult("gp_direct", seed, generations, population_size,
                           best, time.perf_counter() - start,
                           [r.expression for r in results])


def run_gep_direct(puzzles, generations: int = 20, population_size: int = 30,
                   seed: int = 42, head_length: int = 6,
                   crossover_rate: float = 0.7, mutation_rate: float = 0.1,
                   max_steps: int = 120, timeout_per_puzzle: float = 2.0,
                   budget_per_individual: float = 6.0) -> DirectRunResult:
    start = time.perf_counter()
    rng = np.random.default_rng(seed)

    population = gep_population(rng, population_size, head_length,
                                terminals=DIRECT_TERMINALS)

    def score_all(pop):
        out = []
        for genome in pop:
            tree = decode_genome_to_tree(genome)
            out.append(_evaluate_tree_direct(
                tree, puzzles, max_steps, timeout_per_puzzle,
                budget_per_individual, genome_string=genome.to_string()))
        return out

    results = score_all(population)
    best = max(results, key=lambda r: r.fitness)
    for _generation in range(1, generations + 1):
        fitnesses = [r.fitness for r in results]
        elite = max(range(len(results)), key=lambda i: fitnesses[i])
        new_population = [population[elite].copy()]
        while len(new_population) < population_size:
            child = gep_select(population, fitnesses, rng)
            if rng.random() < crossover_rate:
                other = gep_select(population, fitnesses, rng)
                if rng.random() < 0.5:
                    child = one_point_crossover(child, other, rng)
                else:
                    child = two_point_crossover(child, other, rng)
            child = point_mutation(child, rng, mutation_rate,
                                   terminals=DIRECT_TERMINALS)
            new_population.append(child)
        population = new_population
        results = score_all(population)
        generation_best = max(results, key=lambda r: r.fitness)
        if generation_best.fitness > best.fitness:
            best = generation_best

    return DirectRunResult("gep_direct", seed, generations, population_size,
                           best, time.perf_counter() - start,
                           [r.expression for r in results])
