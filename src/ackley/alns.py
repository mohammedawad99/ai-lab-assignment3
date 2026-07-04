"""ALNS-style search on the Ackley function.

ALNS is designed for combinatorial problems, so this is a simple continuous
adaptation for the warm-up: "destroy" picks a subset of dimensions and
"repair" assigns them new values. Operator weights adapt like in the CVRP
version. This is a practical warm-up variant, not a canonical ALNS.
"""

import math
import time

import numpy as np

from src.ackley.function import ackley_value, clip_to_bounds, sample_uniform
from src.ackley.result import AckleyRunResult

WEIGHT_FLOOR = 0.05

SCORE_NEW_BEST = 5.0
SCORE_IMPROVED = 2.0
SCORE_ACCEPTED = 0.5
SCORE_REJECTED = 0.1


# ---------- small helpers ----------

def choose_by_weight(weights: dict[str, float], rng) -> str:
    names = sorted(weights)
    values = np.array([weights[name] for name in names])
    probs = values / values.sum()
    return names[int(rng.choice(len(names), p=probs))]


def update_weight(weights: dict[str, float], name: str, score: float,
                  reaction_rate: float = 0.2) -> None:
    new_weight = (1.0 - reaction_rate) * weights[name] + reaction_rate * score
    weights[name] = max(WEIGHT_FLOOR, new_weight)


def _destroy_count(dimension, rng, min_destroy, max_destroy_fraction):
    max_destroy = max(min_destroy, int(max_destroy_fraction * dimension))
    max_destroy = min(max_destroy, dimension)
    return int(rng.integers(min_destroy, max_destroy + 1))


# ---------- destroy operators ----------

def random_dimension_destroy(current: list[float], rng, min_destroy: int = 1,
                             max_destroy_fraction: float = 0.4) -> list[int]:
    """A uniformly random subset of dimension indices."""
    n = len(current)
    count = _destroy_count(n, rng, min_destroy, max_destroy_fraction)
    return sorted(int(i) for i in rng.choice(n, size=count, replace=False))


def worst_dimension_destroy(current: list[float], rng, min_destroy: int = 1,
                            max_destroy_fraction: float = 0.4) -> list[int]:
    """Prefer dimensions far from zero (the known optimum), with a bit of
    randomness: pick from the top count+2 largest absolute values."""
    n = len(current)
    count = _destroy_count(n, rng, min_destroy, max_destroy_fraction)
    order = sorted(range(n), key=lambda i: -abs(current[i]))
    pool = order[:min(n, count + 2)]
    return sorted(int(i) for i in rng.choice(pool, size=min(count, len(pool)),
                                             replace=False))


# ---------- repair operators ----------

def random_repair(current: list[float], destroyed_indices: list[int], bounds,
                  rng) -> list[float]:
    """Resample destroyed dimensions uniformly inside their bounds."""
    point = list(current)
    for i in destroyed_indices:
        lo, hi = bounds[i]
        point[i] = float(rng.uniform(lo, hi))
    return clip_to_bounds(point, bounds)


def gaussian_repair(current: list[float], destroyed_indices: list[int], bounds,
                    rng, mutation_scale: float = 1.0) -> list[float]:
    """Add Gaussian noise to destroyed dimensions."""
    point = list(current)
    for i in destroyed_indices:
        point[i] += float(rng.normal(0.0, mutation_scale))
    return clip_to_bounds(point, bounds)


def toward_zero_repair(current: list[float], destroyed_indices: list[int], bounds,
                       rng) -> list[float]:
    """Move destroyed dimensions partway toward zero."""
    point = list(current)
    for i in destroyed_indices:
        point[i] *= float(rng.uniform(0.0, 0.8))
    return clip_to_bounds(point, bounds)


# ---------- full run ----------

def run_ackley_alns(dimension: int = 10, iterations: int = 1000, seed: int = 42,
                    lower: float = -32.768, upper: float = 32.768,
                    timeout_sec: float = 10.0, min_destroy: int = 1,
                    max_destroy_fraction: float = 0.4,
                    initial_temperature: float = 10.0, cooling_rate: float = 0.995,
                    reaction_rate: float = 0.2,
                    mutation_scale: float = 1.0) -> AckleyRunResult:
    start_elapsed = time.perf_counter()
    start_cpu = time.process_time()

    rng = np.random.default_rng(seed)
    bounds = [(lower, upper)] * dimension

    current = sample_uniform(rng, bounds)
    current_value = ackley_value(current)
    best, best_value = list(current), current_value

    destroy_weights = {"random_dimension": 1.0, "worst_dimension": 1.0}
    repair_weights = {"random_repair": 1.0, "gaussian_repair": 1.0,
                      "toward_zero_repair": 1.0}
    temperature = initial_temperature

    completed = 0
    for it in range(1, iterations + 1):
        if time.perf_counter() - start_elapsed > timeout_sec:
            break
        completed = it

        destroy_name = choose_by_weight(destroy_weights, rng)
        repair_name = choose_by_weight(repair_weights, rng)

        if destroy_name == "random_dimension":
            destroyed = random_dimension_destroy(current, rng, min_destroy,
                                                 max_destroy_fraction)
        else:
            destroyed = worst_dimension_destroy(current, rng, min_destroy,
                                                max_destroy_fraction)

        if repair_name == "random_repair":
            candidate = random_repair(current, destroyed, bounds, rng)
        elif repair_name == "gaussian_repair":
            candidate = gaussian_repair(current, destroyed, bounds, rng,
                                        mutation_scale=mutation_scale)
        else:
            candidate = toward_zero_repair(current, destroyed, bounds, rng)

        value = ackley_value(candidate)
        delta = value - current_value
        score = SCORE_REJECTED
        if delta < 0:
            current, current_value = candidate, value
            score = SCORE_IMPROVED
            if current_value < best_value:
                best, best_value = list(current), current_value
                score = SCORE_NEW_BEST
        elif temperature > 1e-12 and rng.random() < math.exp(-delta / temperature):
            current, current_value = candidate, value
            score = SCORE_ACCEPTED

        update_weight(destroy_weights, destroy_name, score, reaction_rate)
        update_weight(repair_weights, repair_name, score, reaction_rate)
        temperature = max(temperature * cooling_rate, 1e-12)

    return AckleyRunResult(
        algorithm="ackley_alns",
        dimension=dimension,
        seed=seed,
        iterations=completed,
        best_value=best_value,
        best_x=best,
        elapsed_time=time.perf_counter() - start_elapsed,
        cpu_time=time.process_time() - start_cpu,
    )
