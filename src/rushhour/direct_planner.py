"""Greedy policy rollout for Rush Hour — the no-A* direct planner bonus.

At every step the planner scores each legal move with a policy function and
applies the best-scoring one (LOWER score = better move, like a heuristic
estimate of remaining difficulty). There is no A*, no open list, no f=g+h
priority queue and no backtracking — a policy either walks to the exit
within max_steps or it does not, which is exactly what the bonus asks to
measure.

Cycle handling: the rollout keeps a visited-state set and prefers the best
move that leads to an unvisited state; when every move is a revisit it
falls back to the overall best move and counts it in repeated_states.
"""

import time
from dataclasses import dataclass, field

from src.rushhour.model import Move, RushHourState
from src.rushhour.moves import apply_move, is_goal, legal_moves


@dataclass
class DirectRolloutResult:
    solved: bool
    steps: int
    repeated_states: int
    dead_end: bool
    runtime_seconds: float
    final_state: RushHourState
    moves: list[Move] = field(default_factory=list)
    error: str | None = None
    stopped_reason: str = ""  # "solved", "max_steps", "dead_end", "timeout", "error"


def greedy_policy_rollout(initial_state: RushHourState, score_move,
                          max_steps: int = 120,
                          timeout_sec: float | None = None) -> DirectRolloutResult:
    """Roll out a policy from initial_state.

    score_move(state, move, after_state, context) must return a float
    (lower = better). context is {"visited": set of board strings,
    "step": int}. The input state is never mutated (states are frozen and
    apply_move returns new ones).
    """
    start = time.perf_counter()
    state = initial_state
    visited = {state.board}
    moves_taken: list[Move] = []
    repeated = 0

    if is_goal(state):
        return DirectRolloutResult(True, 0, 0, False,
                                   time.perf_counter() - start, state,
                                   stopped_reason="solved")

    for step in range(1, max_steps + 1):
        if timeout_sec is not None and time.perf_counter() - start > timeout_sec:
            return DirectRolloutResult(False, step - 1, repeated, False,
                                       time.perf_counter() - start, state,
                                       moves_taken, stopped_reason="timeout")
        candidates = legal_moves(state)
        if not candidates:
            return DirectRolloutResult(False, step - 1, repeated, True,
                                       time.perf_counter() - start, state,
                                       moves_taken, stopped_reason="dead_end")

        context = {"visited": visited, "step": step}
        scored = []
        for move in candidates:
            after = apply_move(state, move)
            try:
                score = float(score_move(state, move, after, context))
            except Exception as exc:  # a broken policy must not crash the run
                return DirectRolloutResult(False, step - 1, repeated, False,
                                           time.perf_counter() - start, state,
                                           moves_taken, error=str(exc),
                                           stopped_reason="error")
            # deterministic tie-break: score, then move identity
            scored.append((score, move.car, move.direction, move.steps,
                           move, after))
        scored.sort(key=lambda item: item[:4])

        # prefer the best move that reaches an unvisited state
        chosen = next(((move, after) for _s, _c, _d, _n, move, after in scored
                       if after.board not in visited), None)
        if chosen is None:
            _s, _c, _d, _n, move, after = scored[0]  # cycle fallback
            chosen = (move, after)
            repeated += 1
        move, state = chosen
        visited.add(state.board)
        moves_taken.append(move)

        if is_goal(state):
            return DirectRolloutResult(True, step, repeated, False,
                                       time.perf_counter() - start, state,
                                       moves_taken, stopped_reason="solved")

    return DirectRolloutResult(False, max_steps, repeated, False,
                               time.perf_counter() - start, state,
                               moves_taken, stopped_reason="max_steps")


# ---------- simple direct baselines (same rollout, no A*) ----------

def random_policy(rng):
    """Score moves with seeded random noise: a pure random legal walk."""
    def score(_state, _move, _after, _context):
        return float(rng.random())
    return score


def greedy_blocker_depth_policy(_state, _move, after, _context):
    """Prefer the move whose resulting board has the lowest blocker depth."""
    from src.rushhour.heuristics import blocker_depth_heuristic
    return float(blocker_depth_heuristic(after))


def greedy_red_distance_policy(_state, _move, after, _context):
    """Prefer the move whose resulting board has the red car closest to exit."""
    from src.rushhour.features import red_car_distance_to_exit
    return float(red_car_distance_to_exit(after))
