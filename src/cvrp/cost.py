"""Cost of routes and full solutions.

Malformed routes are not fixed here; feasibility is checked in validate.py.
"""


def route_cost(route: list[int], distance_matrix: list[list[float]]) -> float:
    """Total travel distance along a route, including depot legs. [0, 0] costs 0."""
    total = 0.0
    for a, b in zip(route, route[1:]):
        total += distance_matrix[a][b]
    return total


def solution_cost(solution, distance_matrix: list[list[float]]) -> float:
    return sum(route_cost(route, distance_matrix) for route in solution.routes)
