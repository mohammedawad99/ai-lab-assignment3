"""Simple board features. These will later be used as GP/GEP terminals."""

from src.rushhour.model import BOARD_SIZE, EMPTY, RED_CAR, RushHourState
from src.rushhour.parser import find_vehicles


def _red_row_and_right_col(state: RushHourState) -> tuple[int, int]:
    red = find_vehicles(state)[RED_CAR]
    row = red.cells[-1] // BOARD_SIZE
    right_col = red.cells[-1] % BOARD_SIZE
    return row, right_col


def red_car_distance_to_exit(state: RushHourState) -> int:
    """Number of cells between the right end of X and the board edge."""
    _, right_col = _red_row_and_right_col(state)
    return BOARD_SIZE - 1 - right_col


def blocking_cars(state: RushHourState) -> int:
    """Number of distinct vehicles on the exit row to the right of X."""
    row, right_col = _red_row_and_right_col(state)
    blockers = set()
    for col in range(right_col + 1, BOARD_SIZE):
        ch = state.board[row * BOARD_SIZE + col]
        if ch != EMPTY:
            blockers.add(ch)
    return len(blockers)


def free_exit_cells(state: RushHourState) -> int:
    """Number of empty cells to the right of X before the first blocker or the edge."""
    row, right_col = _red_row_and_right_col(state)
    count = 0
    for col in range(right_col + 1, BOARD_SIZE):
        if state.board[row * BOARD_SIZE + col] != EMPTY:
            break
        count += 1
    return count
