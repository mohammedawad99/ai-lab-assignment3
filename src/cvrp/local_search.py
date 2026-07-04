"""Local improvement moves for CVRP: in-route 2-opt, cross-route relocate,
and a vehicle-count feasibility repair (Clarke-Wright can produce more routes
than the instance allows, which showed up on the official P-n16-k8)."""

from src.cvrp.construction import is_used_route, route_load
from src.cvrp.cost import route_cost, solution_cost
from src.cvrp.model import CVRPInstance, CVRPSolution
from src.cvrp.validate import validate_solution

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


# ---------- vehicle-count feasibility repair ----------

def used_route_count(solution_or_routes) -> int:
    """Number of routes with at least one customer. [0, 0] is unused."""
    if isinstance(solution_or_routes, CVRPSolution):
        routes = solution_or_routes.routes
    else:
        routes = solution_or_routes
    return sum(1 for route in routes if is_used_route(route))


def route_customers(route: list[int]) -> list[int]:
    return [node for node in route[1:-1] if node != 0]


def insertion_delta(route: list[int], customer: int, position: int,
                    distance_matrix) -> float:
    """Added cost of inserting a customer before route[position]."""
    prev, nxt = route[position - 1], route[position]
    return (distance_matrix[prev][customer] + distance_matrix[customer][nxt]
            - distance_matrix[prev][nxt])


def try_insert_customer_best(instance: CVRPInstance, routes: list[list[int]],
                             customer: int, distance_matrix):
    """Insert the customer at its cheapest capacity-feasible position.
    Returns (new_routes, added_cost) or None if nothing fits."""
    demand = instance.demands.get(customer, 0.0)
    best = None  # (delta, route_idx, pos)
    for r_idx, route in enumerate(routes):
        if route_load(instance, route) + demand > instance.capacity:
            continue
        for pos in range(1, len(route)):
            delta = insertion_delta(route, customer, pos, distance_matrix)
            if best is None or delta < best[0]:
                best = (delta, r_idx, pos)
    if best is None:
        return None
    delta, r_idx, pos = best
    new_routes = [list(route) for route in routes]
    new_routes[r_idx].insert(pos, customer)
    return new_routes, delta


def reduce_route_count(instance: CVRPInstance, solution: CVRPSolution,
                       distance_matrix):
    """Empty small routes and reinsert their customers elsewhere until the
    used route count fits the vehicle limit. Returns a solution or None."""
    routes = [list(route) for route in solution.routes if is_used_route(route)]
    if len(routes) <= instance.vehicle_count:
        copy = CVRPSolution(routes=[list(route) for route in solution.routes])
        copy.cost = solution_cost(copy, distance_matrix)
        return copy

    while len(routes) > instance.vehicle_count:
        # try the smallest routes first: fewest customers, then lowest load
        order = sorted(range(len(routes)),
                       key=lambda i: (len(route_customers(routes[i])),
                                      route_load(instance, routes[i])))
        removed = False
        for idx in order:
            customers = route_customers(routes[idx])
            remaining = [list(route) for j, route in enumerate(routes) if j != idx]
            all_inserted = True
            # reinsert the hardest (largest demand) customers first
            for customer in sorted(customers,
                                   key=lambda c: -instance.demands.get(c, 0.0)):
                result = try_insert_customer_best(instance, remaining, customer,
                                                  distance_matrix)
                if result is None:
                    all_inserted = False
                    break
                remaining, _ = result
            if all_inserted:
                routes = remaining
                removed = True
                break
        if not removed:
            return None  # no route could be emptied any more

    reduced = CVRPSolution(routes=routes)
    reduced.cost = solution_cost(reduced, distance_matrix)
    return reduced


def _rebuild_by_cheapest_insertion(instance: CVRPInstance, customer_order,
                                   distance_matrix):
    """Build routes from scratch by cheapest insertion, never opening more
    than vehicle_count routes. Returns a solution or None."""
    d = distance_matrix
    routes: list[list[int]] = []
    for customer in customer_order:
        demand = instance.demands.get(customer, 0.0)
        if demand > instance.capacity:
            return None
        best = try_insert_customer_best(instance, routes, customer, d)
        new_route_delta = d[0][customer] + d[customer][0]
        can_open = len(routes) < instance.vehicle_count
        if best is not None and (not can_open or best[1] <= new_route_delta):
            routes, _ = best
        elif can_open:
            routes.append([0, customer, 0])
        else:
            return None
    rebuilt = CVRPSolution(routes=routes)
    rebuilt.cost = solution_cost(rebuilt, d)
    return rebuilt


def make_solution_vehicle_feasible(instance: CVRPInstance, solution: CVRPSolution,
                                   distance_matrix):
    """Repair a solution that uses too many routes.

    First tries emptying routes (reduce_route_count); if that fails, rebuilds
    from scratch with a few deterministic customer orders. Returns a validated
    feasible solution or None when no repair works.
    """
    if used_route_count(solution) <= instance.vehicle_count:
        copy = CVRPSolution(routes=[list(route) for route in solution.routes])
        copy.cost = solution_cost(copy, distance_matrix)
        return copy

    def finish(candidate):
        improved = improve_solution_2opt(candidate, distance_matrix)
        if validate_solution(instance, improved).feasible:
            return improved
        return None

    reduced = reduce_route_count(instance, solution, distance_matrix)
    if reduced is not None:
        finished = finish(reduced)
        if finished is not None:
            return finished

    customers = [c for route in solution.routes for c in route_customers(route)]
    orders = [
        list(customers),                                                # solution order
        sorted(customers, key=lambda c: -instance.demands.get(c, 0.0)),  # big demand first
        sorted(customers, key=lambda c: -distance_matrix[0][c]),         # far from depot first
        sorted(customers),                                               # by id
    ]
    for order in orders:
        rebuilt = _rebuild_by_cheapest_insertion(instance, order, distance_matrix)
        if rebuilt is not None:
            finished = finish(rebuilt)
            if finished is not None:
                return finished
    return None
