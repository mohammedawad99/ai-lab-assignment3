"""Simulated Annealing on the Ackley function."""

import math
import time

import numpy as np

from src.ackley.function import ackley_value, clip_to_bounds, sample_uniform
from src.ackley.result import AckleyRunResult


def run_ackley_simulated_annealing(dimension: int = 10, iterations: int = 1000,
                                   seed: int = 42, lower: float = -32.768,
                                   upper: float = 32.768,
                                   initial_temperature: float = 10.0,
                                   cooling_rate: float = 0.995,
                                   step_scale: float = 1.0,
                                   timeout_sec: float = 10.0) -> AckleyRunResult:
    start_elapsed = time.perf_counter()
    start_cpu = time.process_time()

    rng = np.random.default_rng(seed)
    bounds = [(lower, upper)] * dimension

    current = sample_uniform(rng, bounds)
    current_value = ackley_value(current)
    best, best_value = list(current), current_value

    temperature = initial_temperature
    completed = 0
    for it in range(1, iterations + 1):
        if time.perf_counter() - start_elapsed > timeout_sec:
            break
        completed = it

        noise = rng.normal(0.0, step_scale, size=dimension)
        candidate = clip_to_bounds(np.asarray(current) + noise, bounds)
        value = ackley_value(candidate)
        delta = value - current_value
        if delta <= 0 or (temperature > 1e-12
                          and rng.random() < math.exp(-delta / temperature)):
            current, current_value = candidate, value
        if current_value < best_value:
            best, best_value = list(current), current_value

        temperature *= cooling_rate

    return AckleyRunResult(
        algorithm="ackley_sa",
        dimension=dimension,
        seed=seed,
        iterations=completed,
        best_value=best_value,
        best_x=best,
        elapsed_time=time.perf_counter() - start_elapsed,
        cpu_time=time.process_time() - start_cpu,
    )
