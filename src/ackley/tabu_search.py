"""Tabu Search on the Ackley function.

A simple continuous approximation for the warm-up: points are made tabu by
rounding their coordinates to one decimal place.
"""

import time

import numpy as np

from src.ackley.function import ackley_value, clip_to_bounds, sample_uniform
from src.ackley.result import AckleyRunResult


def _signature(point) -> tuple:
    return tuple(round(v, 1) for v in point)


def run_ackley_tabu_search(dimension: int = 10, iterations: int = 1000,
                           seed: int = 42, lower: float = -32.768,
                           upper: float = 32.768, step_scale: float = 1.0,
                           tabu_tenure: int = 30,
                           candidates_per_iteration: int = 20,
                           timeout_sec: float = 10.0) -> AckleyRunResult:
    start_elapsed = time.perf_counter()
    start_cpu = time.process_time()

    rng = np.random.default_rng(seed)
    bounds = [(lower, upper)] * dimension

    current = sample_uniform(rng, bounds)
    best = list(current)
    best_value = ackley_value(current)

    tabu_list: list[tuple] = []  # FIFO, newest at the end
    completed = 0
    for it in range(1, iterations + 1):
        if time.perf_counter() - start_elapsed > timeout_sec:
            break
        completed = it

        candidates = []
        for _ in range(candidates_per_iteration):
            noise = rng.normal(0.0, step_scale, size=dimension)
            point = clip_to_bounds(np.asarray(current) + noise, bounds)
            candidates.append((ackley_value(point), point))
        candidates.sort(key=lambda pair: pair[0])

        chosen = None
        for value, point in candidates:
            aspiration = value < best_value
            if _signature(point) not in tabu_list or aspiration:
                chosen = (value, point)
                break
        if chosen is None:
            continue

        value, point = chosen
        current = point
        tabu_list.append(_signature(point))
        if len(tabu_list) > tabu_tenure:
            tabu_list.pop(0)
        if value < best_value:
            best, best_value = list(point), value

    return AckleyRunResult(
        algorithm="ackley_tabu",
        dimension=dimension,
        seed=seed,
        iterations=completed,
        best_value=best_value,
        best_x=best,
        elapsed_time=time.perf_counter() - start_elapsed,
        cpu_time=time.process_time() - start_cpu,
    )
