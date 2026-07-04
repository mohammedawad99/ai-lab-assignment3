"""Tests for the vehicle-count feasibility repair and the official baseline."""

from pathlib import Path

import pytest

from src.cvrp.baseline import build_multistage_baseline
from src.cvrp.distance import build_distance_matrix
from src.cvrp.io_cvrplib import parse_cvrplib
from src.cvrp.local_search import make_solution_vehicle_feasible, used_route_count
from src.cvrp.model import CVRPInstance, CVRPSolution
from src.cvrp.validate import validate_solution

REPO_ROOT = Path(__file__).resolve().parents[1]
TINY_INSTANCE = REPO_ROOT / "examples" / "tiny_cvrp.vrp"
OFFICIAL_P16 = REPO_ROOT / "data" / "official_cvrp" / "P-n16-k8.vrp"


@pytest.fixture
def synthetic():
    # 4 customers with small demands but only 2 vehicles
    coordinates = {0: (0, 0), 1: (10, 0), 2: (0, 10), 3: (10, 10), 4: (20, 10)}
    demands = {0: 0, 1: 2, 2: 2, 3: 2, 4: 2}
    return CVRPInstance(name="synthetic", capacity=10, vehicle_count=2,
                        coordinates=coordinates, demands=demands)


def test_repair_reduces_route_count(synthetic):
    matrix = build_distance_matrix(synthetic)
    # three used routes but only two vehicles allowed
    solution = CVRPSolution(routes=[[0, 1, 0], [0, 2, 0], [0, 3, 4, 0]])
    assert used_route_count(solution) == 3

    repaired = make_solution_vehicle_feasible(synthetic, solution, matrix)
    assert repaired is not None
    assert used_route_count(repaired) <= synthetic.vehicle_count
    # every customer exactly once
    visited = sorted(n for route in repaired.routes for n in route if n != 0)
    assert visited == [1, 2, 3, 4]
    # capacity respected (validation checks it too)
    check = validate_solution(synthetic, repaired)
    assert check.feasible, check.errors


def test_repair_keeps_already_feasible_solution(synthetic):
    matrix = build_distance_matrix(synthetic)
    solution = CVRPSolution(routes=[[0, 1, 3, 0], [0, 2, 4, 0]])
    repaired = make_solution_vehicle_feasible(synthetic, solution, matrix)
    assert repaired is not None
    assert repaired.routes == solution.routes
    assert used_route_count(repaired) == 2


def test_repair_returns_none_when_impossible():
    # 3 customers with demand 5, capacity 5, only 2 vehicles: cannot fit
    instance = CVRPInstance(
        name="impossible", capacity=5, vehicle_count=2,
        coordinates={0: (0, 0), 1: (1, 0), 2: (2, 0), 3: (3, 0)},
        demands={0: 0, 1: 5, 2: 5, 3: 5},
    )
    matrix = build_distance_matrix(instance)
    solution = CVRPSolution(routes=[[0, 1, 0], [0, 2, 0], [0, 3, 0]])
    assert make_solution_vehicle_feasible(instance, solution, matrix) is None


def test_baseline_still_feasible_on_tiny():
    instance = parse_cvrplib(TINY_INSTANCE)
    result = build_multistage_baseline(instance)
    assert result.feasible
    assert result.final_cost == pytest.approx(80.64, abs=0.05)


@pytest.mark.skipif(not OFFICIAL_P16.exists(),
                    reason="official P-n16-k8.vrp not placed under data/official_cvrp/")
def test_baseline_feasible_on_official_p16():
    instance = parse_cvrplib(OFFICIAL_P16)
    result = build_multistage_baseline(instance)
    assert result.feasible, result.errors
    assert used_route_count(result.solution) <= instance.vehicle_count
    check = validate_solution(instance, result.solution)
    assert check.feasible, check.errors
