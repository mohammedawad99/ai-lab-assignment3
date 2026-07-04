"""Discretized LDS-style search on the Ackley function.

Each dimension is split into equal-width bins and the search assigns one bin
per dimension, exploring bin centers. Bins are ranked by how close their
center is to zero — a greedy guide that openly uses the fact that the known
Ackley optimum is at the origin, which is fine for a warm-up but would not
transfer to an unknown function. Discrepancy counts how far the search
strays from the greedy bin per dimension. This is a small warm-up variant,
not a continuous Branch and Bound proof.
"""

import time

import numpy as np

from src.ackley.function import ackley_value, clip_to_bounds, sample_uniform
from src.ackley.result import AckleyRunResult

OPTIMUM_EPS = 1e-9  # stop early when best_value is essentially 0


# ---------- helpers ----------

def make_bins(lower: float, upper: float, bins_per_dimension: int) -> list[tuple[float, float]]:
    """Equal-width (lo, hi) intervals covering [lower, upper]."""
    width = (upper - lower) / bins_per_dimension
    return [(lower + i * width, lower + (i + 1) * width)
            for i in range(bins_per_dimension)]


def bin_center(bin_interval: tuple[float, float]) -> float:
    return (bin_interval[0] + bin_interval[1]) / 2.0


def sorted_bin_indices_by_center(bins: list[tuple[float, float]]) -> list[int]:
    """Bin indices ordered by |center|, nearest to zero first."""
    return sorted(range(len(bins)), key=lambda i: (abs(bin_center(bins[i])), i))


def point_from_bin_choices(bin_choices: list[int], bins) -> list[float]:
    return [bin_center(bins[choice]) for choice in bin_choices]


def clip_point_to_bounds(point: list[float], bounds) -> list[float]:
    return clip_to_bounds(point, bounds)


# ---------- solver ----------

def run_ackley_bnb_lds(dimension: int = 10, bins_per_dimension: int = 7,
                       max_discrepancy: int = 3, max_nodes: int = 50000,
                       seed: int = 42, lower: float = -32.768,
                       upper: float = 32.768,
                       timeout_sec: float = 10.0) -> AckleyRunResult:
    start_elapsed = time.perf_counter()
    start_cpu = time.process_time()

    bounds = [(lower, upper)] * dimension
    bins = make_bins(lower, upper, bins_per_dimension)
    ranked_bins = sorted_bin_indices_by_center(bins)

    best_x = None
    best_value = float("inf")
    stats = {"expanded_nodes": 0, "pruned_by_discrepancy": 0, "completed_points": 0}
    stop_reason = None

    def dfs(bin_choices, discrepancy):
        nonlocal best_x, best_value, stop_reason
        if stop_reason is not None:
            return
        if time.perf_counter() - start_elapsed > timeout_sec:
            stop_reason = "timeout"
            return
        if stats["expanded_nodes"] >= max_nodes:
            stop_reason = "max_nodes"
            return
        stats["expanded_nodes"] += 1

        if len(bin_choices) == dimension:
            point = clip_point_to_bounds(point_from_bin_choices(bin_choices, bins), bounds)
            value = ackley_value(point)
            stats["completed_points"] += 1
            if value < best_value:
                best_x, best_value = point, value
            # the Ackley lower bound is 0, so a near-zero best cannot be beaten
            if best_value <= OPTIMUM_EPS:
                stop_reason = "completed"
            return

        for rank, bin_index in enumerate(ranked_bins):
            new_discrepancy = discrepancy + rank
            if new_discrepancy > max_discrepancy:
                stats["pruned_by_discrepancy"] += len(ranked_bins) - rank
                break
            dfs(bin_choices + [bin_index], new_discrepancy)
            if stop_reason is not None:
                return

    dfs([], 0)
    if stop_reason is None:
        stop_reason = "completed"

    if best_x is None:
        # limits were too strict for even one full point: fall back to a random one
        rng = np.random.default_rng(seed)
        best_x = sample_uniform(rng, bounds)
        best_value = ackley_value(best_x)

    return AckleyRunResult(
        algorithm="ackley_bnb_lds",
        dimension=dimension,
        seed=seed,
        iterations=stats["expanded_nodes"],
        best_value=best_value,
        best_x=best_x,
        elapsed_time=time.perf_counter() - start_elapsed,
        cpu_time=time.process_time() - start_cpu,
    )
