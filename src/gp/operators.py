"""Genetic operators for GP trees: crossover, mutation, selection."""

from src.gp.generator import random_tree
from src.gp.tree import GPNode, GPTree


def collect_nodes(root: GPNode) -> list[GPNode]:
    """All nodes in preorder."""
    nodes = [root]
    for child in root.children:
        nodes.extend(collect_nodes(child))
    return nodes


def replace_subtree(root: GPNode, target_index: int, new_subtree: GPNode) -> GPNode:
    """Return a copy of root with the preorder node target_index replaced."""
    if target_index == 0:
        return new_subtree.copy()
    copied = root.copy()
    nodes = collect_nodes(copied)
    target = nodes[target_index]
    for node in nodes:
        for i, child in enumerate(node.children):
            if child is target:
                node.children[i] = new_subtree.copy()
                return copied
    return copied  # should not happen for a valid index


def subtree_crossover(parent1: GPTree, parent2: GPTree, rng,
                      max_depth: int = 6) -> GPTree:
    """Swap a random subtree of parent2 into a random spot of parent1."""
    nodes1 = collect_nodes(parent1.root)
    nodes2 = collect_nodes(parent2.root)
    i = int(rng.integers(0, len(nodes1)))
    j = int(rng.integers(0, len(nodes2)))
    child = GPTree(root=replace_subtree(parent1.root, i, nodes2[j]))
    if child.depth() > max_depth:
        return parent1.copy()  # too deep, keep the parent instead
    return child


def subtree_mutation(parent: GPTree, rng, max_depth: int = 6) -> GPTree:
    """Replace a random subtree with a small random one."""
    nodes = collect_nodes(parent.root)
    i = int(rng.integers(0, len(nodes)))
    new_subtree = random_tree(rng, max_depth=3, grow=True).root
    child = GPTree(root=replace_subtree(parent.root, i, new_subtree))
    if child.depth() > max_depth:
        return parent.copy()
    return child


def tournament_select(population: list[GPTree], fitnesses: list[float], rng,
                      tournament_size: int = 3) -> GPTree:
    """Sample a few individuals and return a copy of the fittest (higher is better)."""
    size = min(tournament_size, len(population))
    picked = rng.choice(len(population), size=size, replace=False)
    best_index = max((int(i) for i in picked), key=lambda i: fitnesses[i])
    return population[best_index].copy()
