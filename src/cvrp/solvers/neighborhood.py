"""Random neighborhood moves shared by SA and Tabu Search.

All functions return a new CVRPSolution and never mutate the input. Moves
keep the depot fixed, never lose or duplicate customers, and respect
capacity, so a feasible input stays feasible. If no valid move is found in a
few tries, a copy of the original solution is returned.
"""

from src.cvrp.construction import route_load
from src.cvrp.cost import solution_cost
from src.cvrp.model import CVRPInstance, CVRPSolution

# how often a move is retried before giving up and returning a copy
TRIES = 20


def clone_routes(routes: list[list[int]]) -> list[list[int]]:
    return [list(route) for route in routes]


def remove_empty_routes(routes: list[list[int]]) -> list[list[int]]:
    """Drop [0, 0] routes. Returns a new list."""
    return [list(route) for route in routes if any(node != 0 for node in route[1:-1])]


def solution_from_routes(routes: list[list[int]], distance_matrix) -> CVRPSolution:
    solution = CVRPSolution(routes=clone_routes(routes))
    solution.cost = solution_cost(solution, distance_matrix)
    return solution


def random_relocate_neighbor(instance: CVRPInstance, solution: CVRPSolution,
                             distance_matrix, rng) -> CVRPSolution:
    """Move one random customer to a random position in another route."""
    routes = solution.routes
    for _ in range(TRIES):
        from_choices = [i for i, r in enumerate(routes) if len(r) > 2]
        if not from_choices or len(routes) < 2:
            break
        a = int(rng.choice(from_choices))
        pos_a = int(rng.integers(1, len(routes[a]) - 1))
        customer = routes[a][pos_a]
        b = int(rng.choice([i for i in range(len(routes)) if i != a]))
        demand = instance.demands.get(customer, 0.0)
        if route_load(instance, routes[b]) + demand > instance.capacity:
            continue
        new_routes = clone_routes(routes)
        new_routes[a].pop(pos_a)
        new_routes[b].insert(int(rng.integers(1, len(new_routes[b]))), customer)
        return solution_from_routes(new_routes, distance_matrix)
    return solution_from_routes(routes, distance_matrix)


def random_swap_neighbor(instance: CVRPInstance, solution: CVRPSolution,
                         distance_matrix, rng) -> CVRPSolution:
    """Swap two random customers (same or different routes)."""
    routes = solution.routes
    positions = [(i, p) for i, r in enumerate(routes) for p in range(1, len(r) - 1)]
    for _ in range(TRIES):
        if len(positions) < 2:
            break
        picked = rng.choice(len(positions), size=2, replace=False)
        (a, pa), (b, pb) = positions[int(picked[0])], positions[int(picked[1])]
        new_routes = clone_routes(routes)
        new_routes[a][pa], new_routes[b][pb] = new_routes[b][pb], new_routes[a][pa]
        if a != b:
            if (route_load(instance, new_routes[a]) > instance.capacity
                    or route_load(instance, new_routes[b]) > instance.capacity):
                continue
        return solution_from_routes(new_routes, distance_matrix)
    return solution_from_routes(routes, distance_matrix)


def random_two_opt_neighbor(instance: CVRPInstance, solution: CVRPSolution,
                            distance_matrix, rng) -> CVRPSolution:
    """Reverse a random inner segment of one route. Capacity is unchanged."""
    routes = solution.routes
    candidates = [i for i, r in enumerate(routes) if len(r) >= 4]  # 2+ customers
    if candidates:
        a = int(rng.choice(candidates))
        route = routes[a]
        i = int(rng.integers(1, len(route) - 2))
        j = int(rng.integers(i + 1, len(route) - 1))
        new_routes = clone_routes(routes)
        new_routes[a] = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
        return solution_from_routes(new_routes, distance_matrix)
    return solution_from_routes(routes, distance_matrix)


def random_neighbor(instance: CVRPInstance, solution: CVRPSolution,
                    distance_matrix, rng) -> CVRPSolution:
    """Apply one randomly chosen operator: relocate, swap or 2-opt."""
    operators = [random_relocate_neighbor, random_swap_neighbor, random_two_opt_neighbor]
    op = operators[int(rng.integers(0, len(operators)))]
    return op(instance, solution, distance_matrix, rng)
