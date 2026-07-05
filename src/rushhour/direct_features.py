"""Move features for the direct GP/GEP Rush Hour planner (Stage 12-B bonus).

Each candidate legal move is described by cheap numeric features of the
board BEFORE and AFTER the move — no A*, no lookahead search, no planning.
These feature names are the terminal set of the direct policy expressions
(see src/gep/symbols.py DIRECT_TERMINALS and direct_fitness.py).
"""

from src.rushhour.features import (
    blocking_cars,
    free_exit_cells,
    red_car_distance_to_exit,
)
from src.rushhour.heuristics import blocker_depth_heuristic
from src.rushhour.model import RED_CAR, Move, RushHourState
from src.rushhour.moves import legal_moves

# feature names, matching src/gep/symbols.py DIRECT_TERMINALS (minus consts)
DIRECT_FEATURE_NAMES = (
    "red_distance", "blockers", "blocker_depth", "free_exit",
    "delta_distance", "delta_blockers", "is_red", "toward_exit",
    "move_steps", "mobility", "visited",
)


def move_features(state: RushHourState, move: Move, after: RushHourState,
                  visited) -> dict:
    """Numeric features of one candidate move.

    All values are deterministic and O(board) to compute; `visited` is the
    set of board strings the rollout has already been in, giving the policy
    a way to penalize moves that walk back into known states.
    """
    distance_before = red_car_distance_to_exit(state)
    blockers_before = blocking_cars(state)
    distance_after = red_car_distance_to_exit(after)
    blockers_after = blocking_cars(after)
    return {
        # board after the move
        "red_distance": float(distance_after),
        "blockers": float(blockers_after),
        "blocker_depth": float(blocker_depth_heuristic(after)),
        "free_exit": float(free_exit_cells(after)),
        # change caused by the move (negative = progress)
        "delta_distance": float(distance_after - distance_before),
        "delta_blockers": float(blockers_after - blockers_before),
        # the move itself
        "is_red": 1.0 if move.car == RED_CAR else 0.0,
        "toward_exit": 1.0 if move.car == RED_CAR and move.direction == "R" else 0.0,
        "move_steps": float(move.steps),
        # how much freedom the position keeps
        "mobility": float(len(legal_moves(after))),
        # cycle signal: 1 when the move returns to an already-seen state
        "visited": 1.0 if after.board in visited else 0.0,
    }
