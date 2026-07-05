"""Random GP tree generation."""

from src.gp.primitives import BINARY_OPERATORS, FEATURE_TERMINALS, UNARY_OPERATORS
from src.gp.tree import GPNode, GPTree

CONSTANTS = (0.0, 0.5, 1.0, 2.0, 5.0)

# probabilities used during generation
P_CONST = 0.25       # terminal is a constant instead of a feature
P_STOP_EARLY = 0.3   # grow-mode chance to stop before max depth
P_UNARY = 0.25       # inner node is unary instead of binary


def random_terminal(rng, terminals=FEATURE_TERMINALS) -> GPNode:
    if rng.random() < P_CONST:
        value = CONSTANTS[int(rng.integers(0, len(CONSTANTS)))]
        return GPNode(kind="const", value=float(value))
    name = terminals[int(rng.integers(0, len(terminals)))]
    return GPNode(kind="terminal", value=name)


def _random_node(rng, depth: int, max_depth: int, grow: bool,
                 terminals=FEATURE_TERMINALS) -> GPNode:
    if depth >= max_depth or (grow and rng.random() < P_STOP_EARLY):
        return random_terminal(rng, terminals)
    if rng.random() < P_UNARY:
        op = UNARY_OPERATORS[int(rng.integers(0, len(UNARY_OPERATORS)))]
        return GPNode(kind="unary", value=op,
                      children=[_random_node(rng, depth + 1, max_depth, grow,
                                             terminals)])
    op = BINARY_OPERATORS[int(rng.integers(0, len(BINARY_OPERATORS)))]
    return GPNode(kind="binary", value=op,
                  children=[_random_node(rng, depth + 1, max_depth, grow, terminals),
                            _random_node(rng, depth + 1, max_depth, grow, terminals)])


def random_tree(rng, max_depth: int = 4, grow: bool = True,
                terminals=FEATURE_TERMINALS) -> GPTree:
    return GPTree(root=_random_node(rng, 1, max_depth, grow, terminals))


def make_initial_population(rng, population_size: int = 20, max_depth: int = 4,
                            terminals=FEATURE_TERMINALS) -> list[GPTree]:
    """Half grow, half full trees (a simple ramped mix)."""
    population = []
    for i in range(population_size):
        population.append(random_tree(rng, max_depth=max_depth,
                                      grow=(i % 2 == 0), terminals=terminals))
    return population
