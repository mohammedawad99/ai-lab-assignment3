"""Baseline heuristics for A*. Simple baselines, not claimed to be admissible."""

from src.rushhour.features import blocking_cars, red_car_distance_to_exit
from src.rushhour.model import BOARD_SIZE, EMPTY, RED_CAR, RushHourState
from src.rushhour.moves import legal_moves
from src.rushhour.parser import find_vehicles


def zero_heuristic(state: RushHourState) -> int:
    return 0


def blocking_heuristic(state: RushHourState) -> int:
    return blocking_cars(state)


def blocking_distance_heuristic(state: RushHourState) -> int:
    return blocking_cars(state) + red_car_distance_to_exit(state)


def blocker_depth_heuristic(state: RushHourState) -> int:
    """Blocking + distance, plus one extra point for every exit-row blocker
    that currently cannot move at all (it must first be freed by another
    car). A stronger manual guide; also not claimed admissible."""
    vehicles = find_vehicles(state)
    red = vehicles[RED_CAR]
    row = red.cells[-1] // BOARD_SIZE
    right_col = red.cells[-1] % BOARD_SIZE
    blockers = set()
    for col in range(right_col + 1, BOARD_SIZE):
        ch = state.board[row * BOARD_SIZE + col]
        if ch != EMPTY:
            blockers.add(ch)
    movable = {move.car for move in legal_moves(state)}
    stuck = sum(1 for name in blockers if name not in movable)
    return len(blockers) + red_car_distance_to_exit(state) + stuck


# names used by the CLI --heuristic flag
BASELINE_HEURISTICS = {
    "zero": zero_heuristic,
    "blocking": blocking_heuristic,
    "blocking_distance": blocking_distance_heuristic,
    "blocker_depth": blocker_depth_heuristic,
}
