"""Karva / K-expression decoder: linear GEP genome -> expression tree.

Symbols are read left to right; the first one is the root, and children are
attached level by level according to each function's arity. Symbols left
over after the expression is complete are simply unused, which is normal in
GEP. The decoded tree reuses the GPNode/GPTree structure only as an
evaluator — the representation and decoding are GEP-specific.
"""

from collections import deque

from src.gep.genome import GEPGenome
from src.gep.symbols import ARITY, is_function
from src.gp.tree import GPNode, GPTree


def _make_node(symbol: str) -> GPNode:
    if is_function(symbol):
        kind = "unary" if ARITY[symbol] == 1 else "binary"
        return GPNode(kind=kind, value=symbol)
    try:
        return GPNode(kind="const", value=float(symbol))
    except ValueError:
        return GPNode(kind="terminal", value=symbol)


def decode_genome_to_tree(genome: GEPGenome) -> GPTree:
    symbols = genome.genes
    root = _make_node(symbols[0])
    open_nodes = deque()
    if ARITY.get(symbols[0], 0) > 0:
        open_nodes.append((root, ARITY[symbols[0]]))

    index = 1
    while open_nodes:
        node, needed = open_nodes[0]
        if len(node.children) == needed:
            open_nodes.popleft()
            continue
        if index < len(symbols):
            symbol = symbols[index]
            index += 1
        else:
            symbol = "0.0"  # genome ended early, pad with a safe terminal
        child = _make_node(symbol)
        node.children.append(child)
        arity = ARITY.get(symbol, 0)
        if arity > 0:
            open_nodes.append((child, arity))

    return GPTree(root=root)


def genome_to_heuristic(genome: GEPGenome):
    """Safe heuristic callable for the decoded genome (no eval, no codegen)."""
    return decode_genome_to_tree(genome).to_heuristic()
