"""Symbol set for GEP genomes.

Same semantic building blocks as GP (Rush Hour features, small constants,
protected operators), but defined separately because GEP works on linear
genomes of symbols, not trees.
"""

TERMINALS = ("distance", "blocking", "free", "0.0", "0.5", "1.0", "2.0", "5.0")
FUNCTIONS = ("+", "-", "*", "/", "min", "max", "abs", "neg", "log")

ARITY = {
    "+": 2, "-": 2, "*": 2, "/": 2, "min": 2, "max": 2,
    "abs": 1, "neg": 1, "log": 1,
    **{terminal: 0 for terminal in TERMINALS},
}
MAX_ARITY = 2

# head symbols: functions or terminals; tail symbols: terminals only
HEAD_SYMBOLS = FUNCTIONS + TERMINALS


def is_terminal(symbol: str) -> bool:
    return symbol in TERMINALS


def is_function(symbol: str) -> bool:
    return symbol in FUNCTIONS


def random_terminal(rng) -> str:
    return TERMINALS[int(rng.integers(0, len(TERMINALS)))]


def random_function(rng) -> str:
    return FUNCTIONS[int(rng.integers(0, len(FUNCTIONS)))]


def random_head_symbol(rng) -> str:
    return HEAD_SYMBOLS[int(rng.integers(0, len(HEAD_SYMBOLS)))]


def random_tail_symbol(rng) -> str:
    return random_terminal(rng)
