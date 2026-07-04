"""Protected primitives and terminals for GP heuristic expressions.

Terminals map to the simple Rush Hour features. All math is protected so an
evolved expression can never divide by zero or take an invalid log. Evolved
heuristics are NOT claimed to be admissible.
"""

import math

from src.rushhour.features import blocking_cars, free_exit_cells, red_car_distance_to_exit

FEATURE_TERMINALS = ("distance", "blocking", "free")
UNARY_OPERATORS = ("abs", "neg", "log")
BINARY_OPERATORS = ("+", "-", "*", "/", "min", "max")

CLAMP_LOWER = -1000.0
CLAMP_UPPER = 1000.0


def protected_div(a: float, b: float) -> float:
    """Division that returns 1.0 when the denominator is near zero."""
    if abs(b) < 1e-9:
        return 1.0
    return a / b


def safe_log_abs(x: float) -> float:
    """log(|x| + 1), safe for zero and negative inputs."""
    return math.log(abs(x) + 1.0)


def clamp_value(x: float, lower: float = CLAMP_LOWER, upper: float = CLAMP_UPPER) -> float:
    if math.isnan(x):
        return 0.0
    return min(max(x, lower), upper)


def terminal_value(name: str, state) -> float:
    """Value of a feature terminal on a Rush Hour state."""
    if name == "distance":
        return float(red_car_distance_to_exit(state))
    if name == "blocking":
        return float(blocking_cars(state))
    if name == "free":
        return float(free_exit_cells(state))
    # "const" terminals carry their own value and are evaluated by the tree node
    raise ValueError(f"unknown terminal '{name}'")


def apply_unary(op: str, a: float) -> float:
    if op == "abs":
        return abs(a)
    if op == "neg":
        return -a
    if op == "log":
        return safe_log_abs(a)
    raise ValueError(f"unknown unary operator '{op}'")


def apply_binary(op: str, a: float, b: float) -> float:
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if op == "/":
        return protected_div(a, b)
    if op == "min":
        return min(a, b)
    if op == "max":
        return max(a, b)
    raise ValueError(f"unknown binary operator '{op}'")
