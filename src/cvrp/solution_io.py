"""Reading and writing CVRP solution files.

Format: first line is the total cost with 2 decimals, then one route per line.
"""

from src.cvrp.cost import solution_cost
from src.cvrp.model import CVRPSolution


def _is_used_route(route: list[int]) -> bool:
    """A used route has at least one customer between the two depot zeros."""
    return any(node != 0 for node in route[1:-1])


def format_solution(solution, distance_matrix, include_unused_vehicles=False,
                    vehicle_count=None) -> str:
    total = solution_cost(solution, distance_matrix)
    lines = [f"{total:.2f}"]
    for route in solution.routes:
        if _is_used_route(route):
            lines.append(" ".join(str(node) for node in route))
    if include_unused_vehicles:
        if vehicle_count is None:
            raise ValueError("vehicle_count is needed when include_unused_vehicles=True")
        while len(lines) - 1 < vehicle_count:
            lines.append("0 0")
    return "\n".join(lines)


def parse_solution_file(path) -> CVRPSolution:
    with open(path) as f:
        lines = [line.strip() for line in f if line.strip()]

    cost = None
    if lines:
        try:
            cost = float(lines[0])
            lines = lines[1:]
        except ValueError:
            # first line is not a cost, treat it as a route line
            pass

    routes = [[int(token) for token in line.split()] for line in lines]
    return CVRPSolution(routes=routes, cost=cost)
