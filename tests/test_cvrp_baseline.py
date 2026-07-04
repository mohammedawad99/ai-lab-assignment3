"""Tests for CVRP construction, local search and the multi-stage baseline."""

import subprocess
import sys
from pathlib import Path

import pytest

from src.cvrp.baseline import build_multistage_baseline
from src.cvrp.construction import (
    clarke_wright_savings,
    is_used_route,
    route_load,
    single_customer_routes,
)
from src.cvrp.cost import route_cost, solution_cost
from src.cvrp.distance import build_distance_matrix
from src.cvrp.io_cvrplib import parse_cvrplib
from src.cvrp.local_search import (
    improve_route_2opt,
    improve_solution_2opt,
    relocate_best_improvement_pass,
)
from src.cvrp.model import CVRPInstance
from src.cvrp.validate import validate_solution

REPO_ROOT = Path(__file__).resolve().parents[1]
TINY_INSTANCE = REPO_ROOT / "examples" / "tiny_cvrp.vrp"


@pytest.fixture
def tiny():
    return parse_cvrplib(TINY_INSTANCE)


# ---------- construction ----------

def test_route_load(tiny):
    assert route_load(tiny, [0, 1, 2, 0]) == pytest.approx(5.0)  # demands 2 + 3
    assert route_load(tiny, [0, 0]) == pytest.approx(0.0)


def test_is_used_route():
    assert is_used_route([0, 1, 0])
    assert not is_used_route([0, 0])


def test_single_customer_routes(tiny):
    routes = single_customer_routes(tiny)
    assert routes == [[0, 1, 0], [0, 2, 0], [0, 3, 0], [0, 4, 0]]


def test_single_customer_routes_rejects_oversized_demand():
    instance = CVRPInstance(
        name="bad",
        capacity=5,
        vehicle_count=1,
        coordinates={0: (0, 0), 1: (1, 0)},
        demands={0: 0, 1: 9},
    )
    with pytest.raises(ValueError):
        single_customer_routes(instance)


def test_clarke_wright_on_tiny(tiny):
    matrix = build_distance_matrix(tiny)
    solution = clarke_wright_savings(tiny, matrix)
    check = validate_solution(tiny, solution)
    assert check.feasible, check.errors
    assert solution.cost == pytest.approx(80.64, abs=0.05)
    assert solution.cost == pytest.approx(solution_cost(solution, matrix))


# ---------- local search ----------

def test_improve_route_2opt_never_worse(tiny):
    matrix = build_distance_matrix(tiny)
    bad_route = [0, 2, 1, 3, 0]  # visiting 2 before 1 makes the route cross itself
    improved = improve_route_2opt(bad_route, matrix)
    assert route_cost(improved, matrix) <= route_cost(bad_route, matrix)
    # the straightened order should be found for this small case
    assert route_cost(improved, matrix) == pytest.approx(
        route_cost([0, 1, 2, 3, 0], matrix)
    )


def test_improve_solution_2opt_feasible(tiny):
    matrix = build_distance_matrix(tiny)
    solution = clarke_wright_savings(tiny, matrix)
    improved = improve_solution_2opt(solution, matrix)
    check = validate_solution(tiny, improved)
    assert check.feasible, check.errors
    assert improved.cost <= solution.cost + 1e-6


def test_relocate_pass_feasible(tiny):
    matrix = build_distance_matrix(tiny)
    solution = clarke_wright_savings(tiny, matrix)
    routes_before = [list(route) for route in solution.routes]
    improved = relocate_best_improvement_pass(tiny, solution, matrix)
    check = validate_solution(tiny, improved)
    assert check.feasible, check.errors
    assert improved.cost <= solution.cost + 1e-6
    # the input solution must not be mutated
    assert solution.routes == routes_before


# ---------- multi-stage baseline ----------

def test_baseline_on_tiny(tiny):
    result = build_multistage_baseline(tiny)
    assert result.feasible
    assert result.errors == []
    assert result.final_cost == pytest.approx(80.64, abs=0.05)
    assert result.construction_cost >= result.after_2opt_cost - 1e-6
    assert result.after_2opt_cost >= result.final_cost - 1e-6
    assert result.final_cost <= result.construction_cost + 1e-6
    assert result.solution.cost == pytest.approx(result.final_cost)


# ---------- CLI ----------

def run_cli(*extra_args):
    script = REPO_ROOT / "scripts" / "run_cvrp_baseline.py"
    return subprocess.run(
        [sys.executable, str(script), "--instance", str(TINY_INSTANCE), *extra_args],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )


def test_cli_baseline_runs():
    proc = run_cli()
    assert proc.returncode == 0
    assert "feasible: yes" in proc.stdout
    assert "final_cost: 80.64" in proc.stdout


def test_cli_include_unused_vehicles(tiny):
    proc = run_cli("--include-unused-vehicles")
    assert proc.returncode == 0
    lines = proc.stdout.splitlines()
    zero_lines = [line for line in lines if line.strip() == "0 0"]
    # tiny baseline uses 2 of the 4 vehicles, so 2 unused lines are printed
    assert len(zero_lines) == tiny.vehicle_count - 2


def test_cli_output_file(tmp_path):
    output = tmp_path / "baseline_solution.txt"
    proc = run_cli("--output", str(output))
    assert proc.returncode == 0
    assert output.exists()
    content = output.read_text().splitlines()
    assert content[0] == "80.64"
    assert len(content) >= 3  # cost line plus two used routes
