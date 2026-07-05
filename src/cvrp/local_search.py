"""Local improvement moves for CVRP: in-route 2-opt, cross-route relocate,
and a vehicle-count feasibility repair (Clarke-Wright can produce more routes
than the instance allows, which showed up on the official P-n16-k8 and, more
severely, on X-n101-k25)."""

import math

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


def two_opt_route_delta(route: list[int], distance_matrix, neighbors=None):
    """Best 2-opt move for a route using O(1) edge deltas.

    Reversing route[i..j] replaces edges (a,b)=(route[i-1],route[i]) and
    (c,d)=(route[j],route[j+1]) with (a,c) and (b,d); only those four edges
    change, so delta = d[a][c] + d[b][d] - d[a][b] - d[c][d].

    Returns (delta, i, j) for the best improving move, or None. Ties keep the
    first (i, j) in scan order, the same order the full version scans in.

    With neighbors (from candidate_lists.build_candidate_lists) only moves
    whose new edge (a, c) connects a to one of its nearest nodes are scanned;
    neighbors=None keeps the exact full neighborhood.
    """
    n = len(route)
    best_delta, best_i, best_j = -EPS, -1, -1
    for i in range(1, n - 2):
        a, b = route[i - 1], route[i]
        row_a, row_b = distance_matrix[a], distance_matrix[b]
        d_ab = row_a[b]
        near_a = None if neighbors is None else neighbors[a]
        for j in range(i + 1, n - 1):
            c, d = route[j], route[j + 1]
            if near_a is not None and c not in near_a:
                continue
            delta = row_a[c] + row_b[d] - d_ab - distance_matrix[c][d]
            if delta < best_delta:
                best_delta, best_i, best_j = delta, i, j
    if best_i < 0:
        return None
    return best_delta, best_i, best_j


def improve_route_2opt(route: list[int], distance_matrix,
                       neighbors=None) -> list[int]:
    """Apply the best improving 2-opt move until no improvement is left.

    Uses O(1) delta evaluation per candidate move (Stage 11-A); the previous
    full-recomputation version is kept as improve_route_2opt_full.
    """
    best = list(route)
    while True:
        move = two_opt_route_delta(best, distance_matrix, neighbors=neighbors)
        if move is None:
            return best
        _, i, j = move
        best[i:j + 1] = best[i:j + 1][::-1]


def improve_route_2opt_full(route: list[int], distance_matrix) -> list[int]:
    """Reference implementation: builds every candidate route and recomputes
    its full cost. Kept for the equivalence tests against improve_route_2opt."""
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
                                   distance_matrix, neighbors=None) -> CVRPSolution:
    """Move single customers between routes while the best move improves the cost.

    Only capacity-feasible moves are considered. The input solution is not mutated.
    With neighbors (candidate_lists.build_candidate_lists) a customer is only
    tried at positions adjacent to one of its nearest nodes or the depot;
    neighbors=None keeps the exact full neighborhood.
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
                near = None
                if neighbors is not None:
                    near = set(neighbors[customer])
                    near.add(instance.depot_id)

                for b, route_b in enumerate(routes):
                    if b == a:
                        continue
                    if route_load(instance, route_b) + demand > instance.capacity:
                        continue
                    for pos_b in range(1, len(route_b)):
                        prev_b, next_b = route_b[pos_b - 1], route_b[pos_b]
                        if near is not None and prev_b not in near and next_b not in near:
                            continue
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

    # last resort: stronger vehicle-limited rebuilds (needed for X-n101-k25,
    # where the packing is too tight for the simple strategies above)
    strong = try_vehicle_limited_rebuilds(instance, solution, distance_matrix)
    if strong is not None:
        finished = finish(strong)
        if finished is not None:
            return finished
    return None


# ---------- stronger vehicle-limited rebuilds ----------

def customer_order_from_solution(solution: CVRPSolution) -> list[int]:
    return [c for route in solution.routes for c in route_customers(route)]


def customer_distance_from_depot(instance, customer, distance_matrix) -> float:
    return distance_matrix[instance.depot_id][customer]


def customer_angle_from_depot(instance, customer) -> float:
    depot_x, depot_y = instance.coordinates[instance.depot_id]
    x, y = instance.coordinates[customer]
    return math.atan2(y - depot_y, x - depot_x)


def _nearest_neighbor_order(instance, customers, distance_matrix) -> list[int]:
    """Order a customer group starting from the depot, always going to the
    nearest unvisited customer (id breaks ties for determinism)."""
    remaining = list(customers)
    ordered = []
    current = instance.depot_id
    while remaining:
        nxt = min(remaining, key=lambda c: (distance_matrix[current][c], c))
        ordered.append(nxt)
        remaining.remove(nxt)
        current = nxt
    return ordered


def route_solution_from_customer_groups(instance, groups, distance_matrix):
    """Turn capacity-feasible customer groups into routes: nearest-neighbor
    order from the depot, then 2-opt per route."""
    routes = []
    for group in groups:
        if not group:
            continue
        ordered = _nearest_neighbor_order(instance, group, distance_matrix)
        routes.append(improve_route_2opt([0] + ordered + [0], distance_matrix))
    solution = CVRPSolution(routes=routes)
    solution.cost = solution_cost(solution, distance_matrix)
    return solution


def build_routes_best_fit_capacity(instance, customer_order):
    """Best-fit packing into at most vehicle_count groups.
    Returns customer groups or None if the packing fails."""
    groups: list[list[int]] = []
    loads: list[float] = []
    for customer in customer_order:
        demand = instance.demands.get(customer, 0.0)
        if demand > instance.capacity:
            return None
        best_index = None
        best_left = None
        for i, load in enumerate(loads):
            left = instance.capacity - load - demand
            if left >= 0 and (best_left is None or left < best_left):
                best_index, best_left = i, left
        if best_index is not None:
            groups[best_index].append(customer)
            loads[best_index] += demand
        elif len(groups) < instance.vehicle_count:
            groups.append([customer])
            loads.append(demand)
        else:
            return None
    return groups


def build_routes_sweep(instance, customer_order, distance_matrix):
    """Sweep-style packing: fill routes sequentially in the given (angle
    sorted) order. Returns a solution or None."""
    groups: list[list[int]] = [[]]
    load = 0.0
    for customer in customer_order:
        demand = instance.demands.get(customer, 0.0)
        if demand > instance.capacity:
            return None
        if load + demand > instance.capacity:
            if len(groups) >= instance.vehicle_count:
                return None
            groups.append([])
            load = 0.0
        groups[-1].append(customer)
        load += demand
    return route_solution_from_customer_groups(instance, groups, distance_matrix)


def build_routes_subset_sum_packing(instance, customer_order):
    """Fill vehicles one at a time as full as possible, using a subset-sum
    table over the integer demands.

    Needed when total demand almost equals total capacity: X-n101-k25 has
    5147 demand against 25 * 206 = 5150 capacity, so almost every route must
    be loaded completely full, which greedy packing cannot find.
    Returns customer groups or None.
    """
    capacity = int(instance.capacity)
    if instance.capacity != capacity:
        return None  # table needs integer capacity
    demands = {}
    for customer in customer_order:
        demand = instance.demands.get(customer, 0.0)
        if demand != int(demand) or demand > capacity:
            return None  # table needs integer demands
        demands[customer] = int(demand)

    remaining = list(customer_order)
    total_left = sum(demands[c] for c in remaining)
    groups = []
    for bins_left in range(instance.vehicle_count, 0, -1):
        if not remaining:
            break
        # everything not in this bin must still fit into the other bins
        lower = max(0, total_left - (bins_left - 1) * capacity)
        if lower > capacity:
            return None
        # subset-sum: one representative customer subset per reachable load
        reachable = {0: []}
        for customer in remaining:
            demand = demands[customer]
            for load, subset in list(reachable.items()):
                new_load = load + demand
                if new_load <= capacity and new_load not in reachable:
                    reachable[new_load] = subset + [customer]
        best_load = max((load for load in reachable if load >= lower), default=None)
        if best_load is None:
            return None
        chosen = reachable[best_load]
        groups.append(chosen)
        chosen_set = set(chosen)
        remaining = [c for c in remaining if c not in chosen_set]
        total_left -= best_load
    if remaining:
        return None
    return groups


def build_routes_regret_insertion(instance, customer_order, distance_matrix):
    """Regret-2 insertion under a hard vehicle limit. Returns a solution or
    None. Slower than the packing strategies, so it is used as a last resort."""
    d = distance_matrix
    routes: list[list[int]] = []
    remaining = list(customer_order)
    while remaining:
        best_pick = None  # (key, customer, route_idx, pos)
        for customer in remaining:
            demand = instance.demands.get(customer, 0.0)
            options = []
            for r_idx, route in enumerate(routes):
                if route_load(instance, route) + demand > instance.capacity:
                    continue
                for pos in range(1, len(route)):
                    options.append((insertion_delta(route, customer, pos, d),
                                    r_idx, pos))
            if demand <= instance.capacity and len(routes) < instance.vehicle_count:
                options.append((d[0][customer] + d[customer][0], None, None))
            if not options:
                return None
            options.sort(key=lambda item: item[0])
            regret = options[1][0] - options[0][0] if len(options) > 1 else 1e9
            # tie-break: larger demand, then farther from depot, then smaller id
            key = (regret, demand, d[instance.depot_id][customer], -customer)
            if best_pick is None or key > best_pick[0]:
                best_pick = (key, customer, options[0][1], options[0][2])
        _, customer, r_idx, pos = best_pick
        if r_idx is None:
            routes.append([0, customer, 0])
        else:
            routes[r_idx].insert(pos, customer)
        remaining.remove(customer)
    solution = CVRPSolution(routes=routes)
    solution.cost = solution_cost(solution, d)
    return solution


def try_vehicle_limited_rebuilds(instance, solution, distance_matrix):
    """Deterministic rebuilds that never exceed the vehicle limit.
    Returns the cheapest feasible candidate, or None."""
    d = distance_matrix
    customers = customer_order_from_solution(solution)

    def demand(c):
        return instance.demands.get(c, 0.0)

    def dist(c):
        return customer_distance_from_depot(instance, c, d)

    angle = lambda c: customer_angle_from_depot(instance, c)

    orders = [
        list(customers),                                      # solution order
        sorted(customers, key=lambda c: -demand(c)),          # big demand first
        sorted(customers, key=lambda c: -dist(c)),            # far from depot first
        sorted(customers, key=lambda c: (-demand(c), -dist(c))),
        sorted(customers),                                    # by id
        sorted(customers, key=angle),                         # angle ascending
        sorted(customers, key=angle, reverse=True),           # angle descending
    ]

    candidates = []

    # A. best-fit capacity packing (cheap, good at tight packing)
    for order in orders:
        groups = build_routes_best_fit_capacity(instance, order)
        if groups is not None:
            candidates.append(route_solution_from_customer_groups(instance, groups, d))

    # C. sweep packing: both directions, a handful of rotations
    by_angle = sorted(customers, key=angle)
    n = len(by_angle)
    step = max(1, n // 8)
    for direction in (by_angle, list(reversed(by_angle))):
        for start in range(0, n, step):
            rotated = direction[start:] + direction[:start]
            swept = build_routes_sweep(instance, rotated, d)
            if swept is not None:
                candidates.append(swept)

    # D. subset-sum packing when the packing is too tight for the above
    # (the group composition depends on the item order, so try a few)
    if not candidates:
        for order in (sorted(customers, key=angle),
                      sorted(customers, key=angle, reverse=True),
                      sorted(customers, key=lambda c: -demand(c)),
                      list(customers)):
            groups = build_routes_subset_sum_packing(instance, order)
            if groups is not None:
                candidates.append(route_solution_from_customer_groups(instance, groups, d))

    # B. regret insertion only when everything else failed
    if not candidates:
        for order in orders:
            built = build_routes_regret_insertion(instance, order, d)
            if built is not None:
                candidates.append(built)

    feasible = [c for c in candidates if validate_solution(instance, c).feasible]
    if not feasible:
        return None
    return min(feasible, key=lambda c: c.cost)
