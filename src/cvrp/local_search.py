"""Local improvement moves for CVRP: in-route 2-opt and cross-route relocate."""

from src.cvrp.construction import is_used_route, route_load
from src.cvrp.cost import route_cost, solution_cost
from src.cvrp.model import CVRPInstance, CVRPSolution

# small tolerance so floating point noise does not count as an improvement
EPS = 1e-9


def two_opt_route(route: list[int], distance_matrix) -> list[list[int]]:
    """All 2-opt neighbours of a route: reverse an inner segment, depots stay fixed."""
    candidates = []
    n = len(route)
    for i in range(1, n - 2):
        for j in range(i + 1, n - 1):
            candidates.append(route[:i] + route[i:j + 1][::-1] + route[j + 1:])
    return candidates


def improve_route_2opt(route: list[int], distance_matrix) -> list[int]:
    """Apply the best improving 2-opt move until no improvement is left."""
    best = list(route)
    best_cost = route_cost(best, distance_matrix)
    while True:
        improved = False
        for candidate in two_opt_route(best, distance_matrix):
            cost = route_cost(candidate, distance_matrix)
            if cost < best_cost - EPS:
                best, best_cost = candidate, cost
                improved = True
        if not improved:
            return best


def improve_solution_2opt(solution: CVRPSolution, distance_matrix) -> CVRPSolution:
    """Run 2-opt inside every used route. Returns a new solution."""
    routes = []
    for route in solution.routes:
        if is_used_route(route):
            routes.append(improve_route_2opt(route, distance_matrix))
        else:
            routes.append(list(route))
    improved = CVRPSolution(routes=routes)
    improved.cost = solution_cost(improved, distance_matrix)
    return improved


def relocate_best_improvement_pass(instance: CVRPInstance, solution: CVRPSolution,
                                   distance_matrix) -> CVRPSolution:
    """Move single customers between routes while the best move improves the cost.

    Only capacity-feasible moves are considered. The input solution is not mutated.
    """
    d = distance_matrix
    routes = [list(route) for route in solution.routes]

    while True:
        best_delta = -EPS
        best_move = None  # (from_route, from_pos, to_route, to_pos)

        for a, route_a in enumerate(routes):
            for pos_a in range(1, len(route_a) - 1):
                customer = route_a[pos_a]
                if customer == instance.depot_id:
                    continue
                demand = instance.demands.get(customer, 0.0)
                prev_a, next_a = route_a[pos_a - 1], route_a[pos_a + 1]
                # cost saved by taking the customer out of route a
                gain = d[prev_a][customer] + d[customer][next_a] - d[prev_a][next_a]

                for b, route_b in enumerate(routes):
                    if b == a:
                        continue
                    if route_load(instance, route_b) + demand > instance.capacity:
                        continue
                    for pos_b in range(1, len(route_b)):
                        prev_b, next_b = route_b[pos_b - 1], route_b[pos_b]
                        extra = d[prev_b][customer] + d[customer][next_b] - d[prev_b][next_b]
                        delta = extra - gain
                        if delta < best_delta:
                            best_delta = delta
                            best_move = (a, pos_a, b, pos_b)

        if best_move is None:
            break
        a, pos_a, b, pos_b = best_move
        customer = routes[a].pop(pos_a)
        routes[b].insert(pos_b, customer)

    improved = CVRPSolution(routes=routes)
    improved.cost = solution_cost(improved, distance_matrix)
    return improved
