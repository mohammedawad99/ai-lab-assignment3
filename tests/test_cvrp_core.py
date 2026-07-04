"""Tests for the CVRP core: distances, costs and validation."""

import math

import pytest

from src.cvrp.cost import route_cost, solution_cost
from src.cvrp.distance import build_distance_matrix, euclidean_distance
from src.cvrp.model import CVRPInstance, CVRPSolution
from src.cvrp.validate import validate_solution


@pytest.fixture
def tiny_instance():
    # geometry sanity example from the assignment
    coordinates = {0: (0, 0), 1: (10, 0), 2: (20, 0), 3: (20, 10), 4: (10, 10)}
    demands = {0: 0, 1: 2, 2: 3, 3: 5, 4: 5}
    return CVRPInstance(
        name="tiny",
        capacity=10,
        vehicle_count=4,
        coordinates=coordinates,
        demands=demands,
    )


def test_euclidean_distance():
    assert euclidean_distance((0, 0), (3, 4)) == pytest.approx(5.0)
    assert euclidean_distance((1, 1), (1, 1)) == pytest.approx(0.0)


def test_distance_matrix_shape_and_values(tiny_instance):
    matrix = build_distance_matrix(tiny_instance)
    n = tiny_instance.node_count
    assert len(matrix) == n
    assert all(len(row) == n for row in matrix)
    assert matrix[0][0] == pytest.approx(0.0)
    assert matrix[0][1] == pytest.approx(10.0)
    assert matrix[1][0] == pytest.approx(10.0)
    assert matrix[3][0] == pytest.approx(math.hypot(20, 10))


def test_tiny_solution_cost(tiny_instance):
    # assignment sanity example: cost should be about 80.64
    matrix = build_distance_matrix(tiny_instance)
    solution = CVRPSolution(routes=[[0, 1, 2, 3, 0], [0, 4, 0]])
    assert solution_cost(solution, matrix) == pytest.approx(80.64, abs=0.05)


def test_empty_route_costs_nothing(tiny_instance):
    matrix = build_distance_matrix(tiny_instance)
    assert route_cost([0, 0], matrix) == pytest.approx(0.0)


def test_feasible_solution_passes(tiny_instance):
    solution = CVRPSolution(routes=[[0, 1, 2, 3, 0], [0, 4, 0]])
    result = validate_solution(tiny_instance, solution)
    assert result.feasible
    assert result.errors == []


def test_duplicate_customer_fails(tiny_instance):
    solution = CVRPSolution(routes=[[0, 1, 2, 3, 0], [0, 4, 0], [0, 1, 0]])
    result = validate_solution(tiny_instance, solution)
    assert not result.feasible
    assert any("visited 2 times" in e for e in result.errors)


def test_missing_customer_fails(tiny_instance):
    solution = CVRPSolution(routes=[[0, 1, 2, 3, 0]])
    result = validate_solution(tiny_instance, solution)
    assert not result.feasible
    assert any("customer 4 is not visited" in e for e in result.errors)


def test_capacity_violation_fails(tiny_instance):
    # one route with all customers: demand 15 > capacity 10
    solution = CVRPSolution(routes=[[0, 1, 2, 3, 4, 0]])
    result = validate_solution(tiny_instance, solution)
    assert not result.feasible
    assert any("exceeds capacity" in e for e in result.errors)


def test_route_must_start_and_end_at_depot(tiny_instance):
    no_start = CVRPSolution(routes=[[1, 2, 3, 0], [0, 4, 0]])
    assert not validate_solution(tiny_instance, no_start).feasible

    no_end = CVRPSolution(routes=[[0, 1, 2, 3], [0, 4, 0]])
    assert not validate_solution(tiny_instance, no_end).feasible


def test_unknown_node_fails(tiny_instance):
    solution = CVRPSolution(routes=[[0, 1, 2, 3, 0], [0, 4, 99, 0]])
    result = validate_solution(tiny_instance, solution)
    assert not result.feasible
    assert any("unknown node id 99" in e for e in result.errors)


def test_unused_route_is_allowed(tiny_instance):
    solution = CVRPSolution(routes=[[0, 1, 2, 3, 0], [0, 4, 0], [0, 0], [0, 0]])
    result = validate_solution(tiny_instance, solution)
    assert result.feasible
    assert result.errors == []
