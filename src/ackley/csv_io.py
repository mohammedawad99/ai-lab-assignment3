"""CSV output for Ackley run results."""

import csv
from pathlib import Path

from src.ackley.result import AckleyRunResult

CSV_HEADER = [
    "algorithm", "dimension", "seed", "iterations", "best_value",
    "distance_from_origin", "elapsed_time", "cpu_time", "best_x",
]


def write_ackley_results(results: list[AckleyRunResult], path) -> None:
    """One CSV row per run. best_x is stored space-separated."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        for r in results:
            writer.writerow([
                r.algorithm, r.dimension, r.seed, r.iterations,
                f"{r.best_value:.6f}", f"{r.distance_from_origin:.6f}",
                f"{r.elapsed_time:.6f}", f"{r.cpu_time:.6f}",
                " ".join(f"{v:.6f}" for v in r.best_x),
            ])
