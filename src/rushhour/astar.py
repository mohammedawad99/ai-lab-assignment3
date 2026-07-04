"""A* search over Rush Hour states with node and time caps."""

import heapq
import itertools
import time

from src.rushhour.heuristics import blocking_distance_heuristic
from src.rushhour.model import Move, RushHourState, SearchResult
from src.rushhour.moves import apply_move, is_goal, legal_moves


def _reconstruct_moves(parents, state) -> list[Move]:
    moves = []
    entry = parents[state]
    while entry is not None:
        parent, move = entry
        moves.append(move)
        entry = parents[parent]
    moves.reverse()
    return moves


def solve_astar(initial_state: RushHourState, heuristic=blocking_distance_heuristic,
                max_nodes: int = 100000, timeout_sec: float = 10.0) -> SearchResult:
    """A* with g = number of moves. Stops safely on max_nodes or timeout."""
    start_elapsed = time.perf_counter()
    start_cpu = time.process_time()

    expanded = 0
    generated = 0

    def make_result(solved, moves, cost, reason):
        return SearchResult(
            solved=solved,
            moves=moves,
            cost=cost,
            expanded_nodes=expanded,
            generated_nodes=generated,
            elapsed_time=time.perf_counter() - start_elapsed,
            cpu_time=time.process_time() - start_cpu,
            stopped_reason=reason,
        )

    # the counter breaks f ties so states are never compared directly in the heap
    counter = itertools.count()
    best_g = {initial_state: 0}
    parents = {initial_state: None}  # state -> (parent_state, move) or None for the root
    open_heap = [(heuristic(initial_state), next(counter), 0, initial_state)]

    while open_heap:
        if time.perf_counter() - start_elapsed > timeout_sec:
            return make_result(False, [], 0, "timeout")

        f, _, g, state = heapq.heappop(open_heap)
        if g > best_g.get(state, float("inf")):
            continue  # stale heap entry, a cheaper path was found later

        if is_goal(state):
            return make_result(True, _reconstruct_moves(parents, state), g, "solved")

        if expanded >= max_nodes:
            return make_result(False, [], 0, "max_nodes")
        expanded += 1

        for move in legal_moves(state):
            child = apply_move(state, move)
            generated += 1
            child_g = g + 1
            if child_g < best_g.get(child, float("inf")):
                best_g[child] = child_g
                parents[child] = (state, move)
                heapq.heappush(open_heap, (child_g + heuristic(child), next(counter), child_g, child))

    return make_result(False, [], 0, "no_solution")
