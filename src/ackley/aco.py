"""Discretized ACO-style search on the Ackley function.

Ackley is continuous, so this is only a simple ACO-style variant for the
warm-up: each dimension is split into equal-width bins, ants pick one bin per
dimension by pheromone weight and sample a point inside it. It is not a
canonical continuous ACO and is not tuned.
"""

import time

import numpy as np

from src.ackley.function import ackley_value, sample_uniform
from src.ackley.result import AckleyRunResult

PHEROMONE_FLOOR = 1e-6


def run_ackley_aco(dimension: int = 10, iterations: int = 100, ants: int = 20,
                   seed: int = 42, lower: float = -32.768, upper: float = 32.768,
                   bins_per_dimension: int = 25, evaporation_rate: float = 0.2,
                   pheromone_deposit: float = 1.0,
                   timeout_sec: float = 10.0) -> AckleyRunResult:
    start_elapsed = time.perf_counter()
    start_cpu = time.process_time()

    rng = np.random.default_rng(seed)
    bin_width = (upper - lower) / bins_per_dimension
    # one pheromone value per (dimension, bin)
    pheromone = np.ones((dimension, bins_per_dimension))

    # a plain uniform sample as the starting best, so best_x is always defined
    best_x = sample_uniform(rng, [(lower, upper)] * dimension)
    best_value = ackley_value(best_x)
    best_bins = None

    completed = 0
    for it in range(1, iterations + 1):
        if time.perf_counter() - start_elapsed > timeout_sec:
            break
        completed = it

        iteration_best_value = None
        iteration_best_bins = None
        for _ in range(ants):
            bins = [
                int(rng.choice(bins_per_dimension, p=pheromone[d] / pheromone[d].sum()))
                for d in range(dimension)
            ]
            point = [
                lower + (bins[d] + rng.uniform()) * bin_width
                for d in range(dimension)
            ]
            value = ackley_value(point)
            if iteration_best_value is None or value < iteration_best_value:
                iteration_best_value = value
                iteration_best_bins = bins
            if value < best_value:
                best_x, best_value, best_bins = list(point), value, list(bins)

        pheromone = np.maximum(pheromone * (1.0 - evaporation_rate), PHEROMONE_FLOOR)
        if iteration_best_bins is not None:
            for d, b in enumerate(iteration_best_bins):
                pheromone[d][b] += pheromone_deposit
        if best_bins is not None:
            # smaller extra deposit on the global best bins
            for d, b in enumerate(best_bins):
                pheromone[d][b] += 0.5 * pheromone_deposit

    return AckleyRunResult(
        algorithm="ackley_aco",
        dimension=dimension,
        seed=seed,
        iterations=completed,
        best_value=best_value,
        best_x=best_x,
        elapsed_time=time.perf_counter() - start_elapsed,
        cpu_time=time.process_time() - start_cpu,
    )
