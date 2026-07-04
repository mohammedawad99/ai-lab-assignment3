"""Plain random search on the Ackley function.

This is only a sanity baseline to check the objective, bounds, seeding and
CLI. It is NOT one of the six required assignment algorithms.
"""

import time

import numpy as np

from src.ackley.function import ackley_value, sample_uniform
from src.ackley.result import AckleyRunResult


def run_random_search(dimension: int = 10, iterations: int = 1000, seed: int = 42,
                      lower: float = -32.768, upper: float = 32.768) -> AckleyRunResult:
    start_elapsed = time.perf_counter()
    start_cpu = time.process_time()

    rng = np.random.default_rng(seed)
    bounds = [(lower, upper)] * dimension

    best_x = sample_uniform(rng, bounds)
    best_value = ackley_value(best_x)
    for _ in range(iterations - 1):
        candidate = sample_uniform(rng, bounds)
        value = ackley_value(candidate)
        if value < best_value:
            best_x = candidate
            best_value = value

    return AckleyRunResult(
        algorithm="random_search",
        dimension=dimension,
        seed=seed,
        iterations=iterations,
        best_value=best_value,
        best_x=best_x,
        elapsed_time=time.perf_counter() - start_elapsed,
        cpu_time=time.process_time() - start_cpu,
    )
