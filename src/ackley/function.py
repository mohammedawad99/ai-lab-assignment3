"""The Ackley benchmark function and simple bounds helpers.

Assignment setting: d = 10, xi in [-32.768, 32.768], a = 20, b = 0.2,
c = 2*pi, global optimum f(0, ..., 0) = 0.
"""

import math

import numpy as np


def ackley_value(x, a: float = 20.0, b: float = 0.2, c: float = 2 * math.pi) -> float:
    """Ackley function value for a point x (list, tuple or numpy array)."""
    arr = np.asarray(x, dtype=float)
    d = arr.size
    if d == 0:
        raise ValueError("x must not be empty")
    sum_sq = float(np.sum(arr ** 2))
    sum_cos = float(np.sum(np.cos(c * arr)))
    return float(
        -a * math.exp(-b * math.sqrt(sum_sq / d))
        - math.exp(sum_cos / d)
        + a
        + math.e
    )


def default_bounds(dimension: int = 10) -> list[tuple[float, float]]:
    """Assignment bounds: [-32.768, 32.768] in every dimension."""
    return [(-32.768, 32.768)] * dimension


def clip_to_bounds(x, bounds) -> list[float]:
    """Clip every coordinate into its [lower, upper] range."""
    return [min(max(float(v), lo), hi) for v, (lo, hi) in zip(x, bounds)]


def is_inside_bounds(x, bounds) -> bool:
    return all(lo <= float(v) <= hi for v, (lo, hi) in zip(x, bounds))


def sample_uniform(rng: np.random.Generator, bounds) -> list[float]:
    """Sample one point uniformly inside the bounds, using the given generator."""
    return [float(rng.uniform(lo, hi)) for lo, hi in bounds]
