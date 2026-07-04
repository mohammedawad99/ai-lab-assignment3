"""Tests for GP primitives, trees and random generation."""

import math

import numpy as np
import pytest

from src.gp.generator import make_initial_population, random_tree
from src.gp.primitives import protected_div, safe_log_abs
from src.gp.tree import GPNode, GPTree
from src.rushhour.parser import parse_board_string

TINY_BOARD = "......" "..A..." "XXA..." "......" "......" "......"


@pytest.fixture
def tiny_state():
    return parse_board_string(TINY_BOARD)


def test_terminal_evaluation(tiny_state):
    assert GPNode("terminal", "distance").evaluate(tiny_state) == pytest.approx(4.0)
    assert GPNode("terminal", "blocking").evaluate(tiny_state) == pytest.approx(1.0)
    assert GPNode("terminal", "free").evaluate(tiny_state) == pytest.approx(0.0)
    assert GPNode("const", 2.0).evaluate(tiny_state) == pytest.approx(2.0)


def test_protected_div():
    assert protected_div(10.0, 2.0) == pytest.approx(5.0)
    assert protected_div(10.0, 0.0) == pytest.approx(1.0)
    assert protected_div(10.0, 1e-12) == pytest.approx(1.0)


def test_safe_log_abs():
    assert safe_log_abs(0.0) == pytest.approx(0.0)
    assert safe_log_abs(-5.0) == pytest.approx(math.log(6.0))
    assert math.isfinite(safe_log_abs(-1e9))


def test_tree_evaluation_and_heuristic(tiny_state):
    # (blocking + distance)
    tree = GPTree(root=GPNode("binary", "+", [
        GPNode("terminal", "blocking"), GPNode("terminal", "distance"),
    ]))
    assert tree.evaluate(tiny_state) == pytest.approx(5.0)

    heuristic = tree.to_heuristic()
    assert callable(heuristic)
    value = heuristic(tiny_state)
    assert math.isfinite(value)
    assert value >= 0.0


def test_heuristic_clamps_negative(tiny_state):
    tree = GPTree(root=GPNode("unary", "neg", [GPNode("terminal", "distance")]))
    assert tree.evaluate(tiny_state) == pytest.approx(-4.0)
    assert tree.to_heuristic()(tiny_state) == 0.0  # clamped to non-negative


def test_tree_copy_is_independent():
    tree = GPTree(root=GPNode("binary", "+", [
        GPNode("terminal", "blocking"), GPNode("const", 1.0),
    ]))
    clone = tree.copy()
    clone.root.children[1].value = 99.0
    assert tree.root.children[1].value == 1.0


def test_tree_size_and_depth():
    tree = GPTree(root=GPNode("binary", "+", [
        GPNode("terminal", "blocking"),
        GPNode("unary", "abs", [GPNode("terminal", "free")]),
    ]))
    assert tree.size() == 4
    assert tree.depth() == 3


def test_to_string():
    tree = GPTree(root=GPNode("binary", "max", [
        GPNode("terminal", "blocking"), GPNode("terminal", "free"),
    ]))
    assert tree.to_string() == "max(blocking, free)"
    plus = GPTree(root=GPNode("binary", "+", [
        GPNode("terminal", "blocking"), GPNode("terminal", "distance"),
    ]))
    assert plus.to_string() == "(blocking + distance)"


def test_random_tree(tiny_state):
    rng = np.random.default_rng(0)
    for _ in range(20):
        tree = random_tree(rng, max_depth=4)
        assert isinstance(tree, GPTree)
        assert tree.depth() <= 4
        assert math.isfinite(tree.to_heuristic()(tiny_state))
        assert tree.to_string()


def test_make_initial_population():
    rng = np.random.default_rng(1)
    population = make_initial_population(rng, population_size=12, max_depth=4)
    assert len(population) == 12
    assert all(isinstance(tree, GPTree) for tree in population)
