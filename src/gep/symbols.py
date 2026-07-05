"""Symbol set for GEP genomes.

Same semantic building blocks as GP (Rush Hour features, small constants,
protected operators), but defined separately because GEP works on linear
genomes of symbols, not trees.
"""

TERMINALS = ("distance", "blocking", "free", "0.0", "0.5", "1.0", "2.0", "5.0")
FUNCTIONS = ("+", "-", "*", "/", "min", "max", "abs", "neg", "log")

# terminal set for the Stage 12-B direct planner: move features (see
# src/rushhour/direct_features.py) plus the same constants. Regular GEP
# never generates these; they are only used when a generator/operator is
# called with terminals=DIRECT_TERMINALS.
DIRECT_TERMINALS = (
    "red_distance", "blockers", "blocker_depth", "free_exit",
    "delta_distance", "delta_blockers", "is_red", "toward_exit",
    "move_steps", "mobility", "visited",
    "0.0", "0.5", "1.0", "2.0", "5.0",
)

ARITY = {
    "+": 2, "-": 2, "*": 2, "/": 2, "min": 2, "max": 2,
    "abs": 1, "neg": 1, "log": 1,
    **{terminal: 0 for terminal in TERMINALS},
    **{terminal: 0 for terminal in DIRECT_TERMINALS},
}
MAX_ARITY = 2

# head symbols: functions or terminals; tail symbols: terminals only
HEAD_SYMBOLS = FUNCTIONS + TERMINALS


def is_terminal(symbol: str) -> bool:
    return symbol in TERMINALS or symbol in DIRECT_TERMINALS


def is_function(symbol: str) -> bool:
    return symbol in FUNCTIONS


def random_terminal(rng, terminals=TERMINALS) -> str:
    return terminals[int(rng.integers(0, len(terminals)))]


def random_function(rng) -> str:
    return FUNCTIONS[int(rng.integers(0, len(FUNCTIONS)))]


def random_head_symbol(rng, terminals=TERMINALS) -> str:
    head = HEAD_SYMBOLS if terminals is TERMINALS else FUNCTIONS + tuple(terminals)
    return head[int(rng.integers(0, len(head)))]


def random_tail_symbol(rng, terminals=TERMINALS) -> str:
    return random_terminal(rng, terminals)
