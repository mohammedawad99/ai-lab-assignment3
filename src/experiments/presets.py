"""Experiment presets: smoke settings now, timing template for later."""

SMOKE_CVRP_ALGORITHMS = ["sa", "tabu", "aco", "ga_island", "alns", "bnb_lds"]
SMOKE_ACKLEY_ALGORITHMS = ["sa", "tabu", "aco", "ga_island", "alns", "bnb_lds"]

# reference methods, not part of the six required algorithms
REFERENCE_CVRP_ALGORITHMS = ["baseline"]
REFERENCE_ACKLEY_ALGORITHMS = ["random_search"]

SMOKE_SEEDS = [42]
SMOKE_BUDGET = 10
SMOKE_TIMEOUT_SEC = 5.0
SMOKE_ACKLEY_DIMENSION = 4


def default_full_timing_plan() -> dict:
    """Suggested per-instance timeouts for the later full experiments.

    This is only a template: smaller instances get shorter timeouts, larger
    ones more time. Nothing here is used for the smoke runs.
    """
    return {
        "P-n16-k8": {"timeout_sec": 30.0},
        "E-n22-k4": {"timeout_sec": 30.0},
        "A-n32-k5": {"timeout_sec": 60.0},
        "A-n80-k10": {"timeout_sec": 120.0},
        "X-n101-k25": {"timeout_sec": 180.0},
        "M-n200-k17": {"timeout_sec": 300.0},
    }
