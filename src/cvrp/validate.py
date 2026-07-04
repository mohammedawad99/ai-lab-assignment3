"""Feasibility validation for CVRP solutions."""

from dataclasses import dataclass

from src.cvrp.model import CVRPInstance, CVRPSolution


@dataclass
class ValidationResult:
    feasible: bool
    errors: list[str]


def validate_solution(instance: CVRPInstance, solution: CVRPSolution) -> ValidationResult:
    """Check a solution against the instance. Never raises for bad input."""
    errors: list[str] = []
    depot = instance.depot_id
    visit_counts: dict[int, int] = {}
    used_routes = 0

    for idx, route in enumerate(solution.routes):
        if len(route) < 2:
            errors.append(f"route {idx} is too short: {route}")
            continue
        if route[0] != depot:
            errors.append(f"route {idx} does not start at the depot: {route}")
        if route[-1] != depot:
            errors.append(f"route {idx} does not end at the depot: {route}")

        load = 0.0
        has_customer = False
        for node in route[1:-1]:
            if node == depot:
                # the depot must not appear in the middle of a route
                errors.append(f"route {idx} visits the depot in the middle: {route}")
                continue
            if node not in instance.coordinates:
                errors.append(f"route {idx} contains unknown node id {node}")
                continue
            has_customer = True
            visit_counts[node] = visit_counts.get(node, 0) + 1
            # depot demand is never added here, only real customers count
            load += instance.demands.get(node, 0.0)
        if load > instance.capacity:
            errors.append(
                f"route {idx} exceeds capacity: load {load} > capacity {instance.capacity}"
            )
        # a [0, 0] route is an unused vehicle and does not count as a used route
        if has_customer:
            used_routes += 1

    for customer in instance.customer_ids:
        count = visit_counts.get(customer, 0)
        if count == 0:
            errors.append(f"customer {customer} is not visited")
        elif count > 1:
            errors.append(f"customer {customer} is visited {count} times")

    if used_routes > instance.vehicle_count:
        errors.append(
            f"{used_routes} routes used but only {instance.vehicle_count} vehicles available"
        )

    return ValidationResult(feasible=(len(errors) == 0), errors=errors)
