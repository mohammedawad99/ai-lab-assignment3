"""Tests for CVRPLIB parsing, solution I/O and the validation CLI."""

import subprocess
import sys
from pathlib import Path

import pytest

from src.cvrp.distance import build_distance_matrix
from src.cvrp.io_cvrplib import parse_cvrplib
from src.cvrp.solution_io import format_solution, parse_solution_file

REPO_ROOT = Path(__file__).resolve().parents[1]
TINY_INSTANCE = REPO_ROOT / "examples" / "tiny_cvrp.vrp"
TINY_SOLUTION = REPO_ROOT / "examples" / "tiny_solution.txt"


@pytest.fixture
def tiny():
    return parse_cvrplib(TINY_INSTANCE)


def test_parse_tiny_instance(tiny):
    assert tiny.name == "tiny-cvrp"
    assert tiny.capacity == 10
    assert tiny.vehicle_count == 4
    assert tiny.node_count == 5


def test_internal_ids(tiny):
    assert tiny.depot_id == 0
    assert tiny.demands[0] == 0
    assert sorted(tiny.customer_ids) == [1, 2, 3, 4]


def test_original_ids_mapping(tiny):
    assert tiny.original_ids is not None
    # depot is node 1 in the file, customers are nodes 2..5
    assert tiny.original_ids[0] == 1
    assert tiny.original_ids == {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}


def test_parse_solution_file():
    solution = parse_solution_file(TINY_SOLUTION)
    assert solution.cost == pytest.approx(80.64)
    assert solution.routes == [[0, 1, 2, 3, 0], [0, 4, 0]]


def test_format_solution_default_skips_unused(tiny):
    matrix = build_distance_matrix(tiny)
    solution = parse_solution_file(TINY_SOLUTION)
    solution.routes.append([0, 0])  # unused vehicle, should not be printed
    lines = format_solution(solution, matrix).splitlines()
    assert lines[0] == "80.64"
    assert lines[1:] == ["0 1 2 3 0", "0 4 0"]


def test_format_solution_with_unused_vehicles(tiny):
    matrix = build_distance_matrix(tiny)
    solution = parse_solution_file(TINY_SOLUTION)
    lines = format_solution(solution, matrix, include_unused_vehicles=True,
                            vehicle_count=tiny.vehicle_count).splitlines()
    assert lines[0] == "80.64"
    assert len(lines) == 1 + tiny.vehicle_count
    assert lines[3] == "0 0"
    assert lines[4] == "0 0"


def test_cli_runs_on_tiny_example():
    script = REPO_ROOT / "scripts" / "validate_cvrp_solution.py"
    proc = subprocess.run(
        [sys.executable, str(script),
         "--instance", str(TINY_INSTANCE),
         "--solution", str(TINY_SOLUTION)],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert proc.returncode == 0
    assert "feasible: yes" in proc.stdout
    assert "80.64" in proc.stdout
