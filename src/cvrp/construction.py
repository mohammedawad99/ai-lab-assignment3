"""Construction heuristics for CVRP: trivial routes and Clarke-Wright savings."""

from src.cvrp.cost import solution_cost
from src.cvrp.model import CVRPInstance, CVRPSolution


def route_load(instance: CVRPInstance, route: list[int]) -> float:
    """Total demand of the customers in a route. The depot adds nothing."""
    return sum(instance.demands.get(node, 0.0) for node in route if node != instance.depot_id)


def is_used_route(route: list[int]) -> bool:
    """A used route has at least one customer between the depot zeros."""
    return any(node != 0 for node in route[1:-1])


def single_customer_routes(instance: CVRPInstance) -> list[list[int]]:
    """One route [0, c, 0] per customer. Raises if a single demand exceeds capacity."""
    routes = []
    for customer in sorted(instance.customer_ids):
        demand = instance.demands.get(customer, 0.0)
        if demand > instance.capacity:
            raise ValueError(
                f"customer {customer} demand {demand} exceeds capacity {instance.capacity}"
            )
        routes.append([instance.depot_id, customer, instance.depot_id])
    return routes


def clarke_wright_savings(instance: CVRPInstance, distance_matrix) -> CVRPSolution:
    """Classic parallel Clarke-Wright savings construction.

    Start with one route per customer and merge route ends by descending
    saving(i, j) = d(0,i) + d(0,j) - d(i,j), while capacity allows it.
    """
    depot = instance.depot_id
    customers = sorted(instance.customer_ids)

    # routes as customer sequences without the depot, e.g. [2, 3]
    sequences: list[list[int] | None] = [[c] for c in customers]
    loads = [instance.demands.get(c, 0.0) for c in customers]
    route_of = {c: idx for idx, c in enumerate(customers)}

    savings = []
    for a in range(len(customers)):
        for b in range(a + 1, len(customers)):
            i, j = customers[a], customers[b]
            s = distance_matrix[depot][i] + distance_matrix[depot][j] - distance_matrix[i][j]
            savings.append((s, i, j))
    # descending saving, then (i, j) so the order is deterministic
    savings.sort(key=lambda item: (-item[0], item[1], item[2]))

    for s, i, j in savings:
        ri, rj = route_of[i], route_of[j]
        if ri == rj:
            continue
        if loads[ri] + loads[rj] > instance.capacity:
            continue
        seq_i, seq_j = sequences[ri], sequences[rj]

        # both i and j must sit at an end of their routes; reverse when needed
        if seq_i[-1] == i and seq_j[0] == j:
            merged = seq_i + seq_j
        elif seq_i[-1] == i and seq_j[-1] == j:
            merged = seq_i + seq_j[::-1]
        elif seq_i[0] == i and seq_j[0] == j:
            merged = seq_i[::-1] + seq_j
        elif seq_i[0] == i and seq_j[-1] == j:
            merged = seq_j + seq_i
        else:
            continue

        sequences[ri] = merged
        loads[ri] += loads[rj]
        sequences[rj] = None
        for c in merged:
            route_of[c] = ri

    routes = [[depot] + seq + [depot] for seq in sequences if seq is not None]
    solution = CVRPSolution(routes=routes)
    solution.cost = solution_cost(solution, distance_matrix)
    return solution
