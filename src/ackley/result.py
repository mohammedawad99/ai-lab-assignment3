"""Result dataclass for Ackley optimizer runs."""

import math
from dataclasses import dataclass


@dataclass
class AckleyRunResult:
    algorithm: str
    dimension: int
    seed: int
    iterations: int
    best_value: float
    best_x: list[float]
    elapsed_time: float
    cpu_time: float

    @property
    def distance_from_origin(self) -> float:
        """Euclidean distance of best_x from the known optimum at the origin."""
        return math.sqrt(sum(v * v for v in self.best_x))
