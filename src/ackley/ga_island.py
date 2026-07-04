"""Genetic Algorithm with an Island Model on the Ackley function.

Continuous vector chromosomes with blend crossover and Gaussian mutation.
Islands evolve separately and exchange their best vectors in a ring.
"""

import time

import numpy as np

from src.ackley.function import ackley_value, clip_to_bounds, sample_uniform
from src.ackley.result import AckleyRunResult


# ---------- representation ----------

def random_vector(dimension: int, bounds, rng) -> list[float]:
    return sample_uniform(rng, bounds)


def blend_crossover(parent1: list[float], parent2: list[float], rng,
                    alpha: float = 0.5) -> list[float]:
    """BLX-alpha: sample each gene in the parents' range widened by alpha."""
    child = []
    for a, b in zip(parent1, parent2):
        low, high = min(a, b), max(a, b)
        spread = high - low
        child.append(float(rng.uniform(low - alpha * spread, high + alpha * spread)))
    return child


def gaussian_mutation(chromosome: list[float], bounds, rng,
                      mutation_rate: float = 0.2,
                      mutation_scale: float = 1.0) -> list[float]:
    child = list(chromosome)
    for i in range(len(child)):
        if rng.random() < mutation_rate:
            child[i] += float(rng.normal(0.0, mutation_scale))
    return clip_to_bounds(child, bounds)


def evaluate_vector(chromosome: list[float]) -> float:
    return ackley_value(chromosome)


def _tournament(population, values, rng, tournament_size: int = 3) -> list[float]:
    size = min(tournament_size, len(population))
    picked = rng.choice(len(population), size=size, replace=False)
    best_index = min((int(i) for i in picked), key=lambda i: values[i])
    return list(population[best_index])


# ---------- island evolution and migration ----------

def evolve_ackley_island(population: list[list[float]], bounds, generations: int,
                         rng, crossover_rate: float = 0.8,
                         mutation_rate: float = 0.2, mutation_scale: float = 1.0,
                         elitism: int = 1):
    """Evolve one island. Returns (new_population, best_vector, best_value)."""
    population = [list(v) for v in population]
    best_vector = None
    best_value = float("inf")

    def note_best(values):
        nonlocal best_vector, best_value
        for vector, value in zip(population, values):
            if value < best_value:
                best_vector, best_value = list(vector), value

    for _ in range(generations):
        values = [evaluate_vector(v) for v in population]
        note_best(values)
        order = sorted(range(len(population)), key=lambda i: values[i])
        new_population = [list(population[i]) for i in order[:elitism]]
        while len(new_population) < len(population):
            parent1 = _tournament(population, values, rng)
            parent2 = _tournament(population, values, rng)
            if rng.random() < crossover_rate:
                child = blend_crossover(parent1, parent2, rng)
            else:
                child = list(parent1)
            child = gaussian_mutation(child, bounds, rng,
                                      mutation_rate=mutation_rate,
                                      mutation_scale=mutation_scale)
            new_population.append(child)
        population = new_population

    values = [evaluate_vector(v) for v in population]
    note_best(values)
    return population, best_vector, best_value


def migrate_ackley_ring(islands: list[list[list[float]]], migrants: int = 1):
    """Each island sends its best vectors to the next island (ring),
    replacing the worst ones there. Island sizes stay fixed."""
    if migrants <= 0 or len(islands) < 2:
        return islands

    all_values = [[evaluate_vector(v) for v in island] for island in islands]
    best_per_island = []
    for island, values in zip(islands, all_values):
        order = sorted(range(len(island)), key=lambda i: values[i])
        best_per_island.append([list(island[i]) for i in order[:migrants]])

    new_islands = [[list(v) for v in island] for island in islands]
    for idx in range(len(islands)):
        target = (idx + 1) % len(islands)
        worst_order = sorted(range(len(islands[target])),
                             key=lambda i: all_values[target][i], reverse=True)
        for m, migrant in enumerate(best_per_island[idx]):
            new_islands[target][worst_order[m]] = list(migrant)
    return new_islands


# ---------- full run ----------

def run_ackley_ga_island(dimension: int = 10, generations: int = 100,
                         population_size: int = 30, islands: int = 4,
                         seed: int = 42, lower: float = -32.768,
                         upper: float = 32.768, timeout_sec: float = 10.0,
                         crossover_rate: float = 0.8, mutation_rate: float = 0.2,
                         mutation_scale: float = 1.0, migration_interval: int = 20,
                         migrants: int = 1) -> AckleyRunResult:
    start_elapsed = time.perf_counter()
    start_cpu = time.process_time()

    bounds = [(lower, upper)] * dimension
    island_rngs = [np.random.default_rng(seed + i) for i in range(islands)]
    populations = [
        [random_vector(dimension, bounds, island_rngs[i]) for _ in range(population_size)]
        for i in range(islands)
    ]

    best_x = list(populations[0][0])
    best_value = evaluate_vector(best_x)

    completed = 0
    for gen in range(1, generations + 1):
        if time.perf_counter() - start_elapsed > timeout_sec:
            break
        completed = gen

        for i in range(islands):
            populations[i], island_best, island_value = evolve_ackley_island(
                populations[i], bounds, 1, island_rngs[i],
                crossover_rate=crossover_rate, mutation_rate=mutation_rate,
                mutation_scale=mutation_scale,
            )
            if island_value < best_value:
                best_x, best_value = list(island_best), island_value

        if migration_interval > 0 and gen % migration_interval == 0:
            populations = migrate_ackley_ring(populations, migrants)

    return AckleyRunResult(
        algorithm="ackley_ga_island",
        dimension=dimension,
        seed=seed,
        iterations=completed,
        best_value=best_value,
        best_x=best_x,
        elapsed_time=time.perf_counter() - start_elapsed,
        cpu_time=time.process_time() - start_cpu,
    )
