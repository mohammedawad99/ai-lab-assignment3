"""Tests for GEP genome structure, Karva decoding and operators."""

import math

import numpy as np
import pytest

from src.gep.decoder import decode_genome_to_tree, genome_to_heuristic
from src.gep.generator import random_genome
from src.gep.genome import GEPGenome
from src.gep.operators import one_point_crossover, point_mutation, two_point_crossover
from src.gep.symbols import is_function, is_terminal
from src.gp.tree import GPTree
from src.rushhour.parser import parse_board_string

TINY_BOARD = "......" "..A..." "XXA..." "......" "......" "......"


def make_genome(head, head_length):
    """Helper: pad the tail with '0.0' to the required length."""
    tail_length = head_length + 1
    genes = list(head) + ["0.0"] * (head_length + tail_length - len(head))
    return GEPGenome(genes=genes, head_length=head_length)


def valid_head_tail(genome):
    head_ok = all(is_function(s) or is_terminal(s)
                  for s in genome.genes[:genome.head_length])
    tail_ok = all(is_terminal(s) for s in genome.genes[genome.head_length:])
    return head_ok and tail_ok and len(genome.genes) == genome.total_length


def test_genome_lengths():
    genome = random_genome(np.random.default_rng(0), head_length=6)
    assert genome.tail_length == 7
    assert genome.total_length == 13
    assert len(genome.genes) == 13


def test_valid_genome_passes():
    genes = ["+", "distance", "blocking", "0.0", "0.5", "1.0", "free"]
    genome = GEPGenome(genes=genes, head_length=3)
    assert genome.head_length == 3
    assert genome.tail_length == 4


def test_function_in_tail_raises():
    genes = ["+", "distance", "blocking", "0.0", "0.5", "1.0", "+"]
    with pytest.raises(ValueError):
        GEPGenome(genes=genes, head_length=3)


def test_random_genome_structure():
    rng = np.random.default_rng(1)
    for _ in range(20):
        genome = random_genome(rng, head_length=5)
        assert valid_head_tail(genome)


def test_decode_simple_genome():
    # K-expression: + distance blocking -> (distance + blocking)
    genome = GEPGenome(genes=["+", "distance", "blocking", "0.0", "0.5", "1.0", "free"],
                       head_length=3)
    tree = decode_genome_to_tree(genome)
    assert isinstance(tree, GPTree)
    assert tree.to_string() == "(distance + blocking)"


def test_decode_uses_level_order():
    # K-expression: + * distance blocking free ... (Karva is level order)
    # root +, children (* and distance), * gets (blocking, free)
    genome = GEPGenome(
        genes=["+", "*", "distance", "blocking", "free", "0.0", "0.5",
               "1.0", "2.0", "5.0", "0.0", "0.5", "1.0"],
        head_length=6,
    )
    tree = decode_genome_to_tree(genome)
    assert tree.to_string() == "((blocking * free) + distance)"


def test_terminal_root_ignores_rest():
    genome = GEPGenome(genes=["blocking", "distance", "free", "0.0", "0.5", "1.0", "2.0"],
                       head_length=3)
    tree = decode_genome_to_tree(genome)
    assert tree.to_string() == "blocking"


def test_genome_to_heuristic():
    state = parse_board_string(TINY_BOARD)
    genome = GEPGenome(genes=["+", "distance", "blocking", "0.0", "0.5", "1.0", "free"],
                       head_length=3)
    heuristic = genome_to_heuristic(genome)
    assert callable(heuristic)
    value = heuristic(state)
    assert math.isfinite(value)
    assert value >= 0.0
    assert value == pytest.approx(5.0)  # distance 4 + blocking 1


def test_point_mutation_keeps_structure():
    rng = np.random.default_rng(2)
    genome = random_genome(rng, head_length=6)
    for _ in range(20):
        mutated = point_mutation(genome, rng, mutation_rate=0.5)
        assert valid_head_tail(mutated)
    # parent untouched
    assert valid_head_tail(genome)


def test_crossovers_keep_structure():
    rng = np.random.default_rng(3)
    p1 = random_genome(rng, head_length=6)
    p2 = random_genome(rng, head_length=6)
    for _ in range(20):
        assert valid_head_tail(one_point_crossover(p1, p2, rng))
        assert valid_head_tail(two_point_crossover(p1, p2, rng))


def test_to_string():
    genome = random_genome(np.random.default_rng(4), head_length=4)
    text = genome.to_string()
    assert text
    assert len(text.split()) == genome.total_length
