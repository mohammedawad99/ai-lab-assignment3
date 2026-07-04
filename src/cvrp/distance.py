"""Euclidean distances and the full distance matrix for a CVRP instance."""

import math


def euclidean_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def build_distance_matrix(instance) -> list[list[float]]:
    """Return matrix[i][j] = Euclidean distance between internal nodes i and j."""
    n = instance.node_count
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = euclidean_distance(instance.coordinates[i], instance.coordinates[j])
            matrix[i][j] = d
            matrix[j][i] = d
    return matrix
