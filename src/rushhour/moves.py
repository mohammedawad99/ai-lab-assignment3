"""Goal test, legal move generation and move application for Rush Hour."""

from src.rushhour.model import BOARD_CELLS, BOARD_SIZE, EMPTY, RED_CAR, Move, RushHourState
from src.rushhour.parser import find_vehicles

# index offset for one step in each direction
_DELTAS = {"L": -1, "R": 1, "U": -BOARD_SIZE, "D": BOARD_SIZE}


def is_goal(state: RushHourState) -> bool:
    """Solved when every cell to the right of the red car up to the edge is empty."""
    vehicles = find_vehicles(state)
    red = vehicles.get(RED_CAR)
    if red is None or red.orientation != "H":
        return False
    row = red.cells[-1] // BOARD_SIZE
    right_col = red.cells[-1] % BOARD_SIZE
    return all(
        state.board[row * BOARD_SIZE + col] == EMPTY
        for col in range(right_col + 1, BOARD_SIZE)
    )


def legal_moves(state: RushHourState) -> list[Move]:
    """All legal multi-step moves, in a fixed deterministic order."""
    vehicles = find_vehicles(state)
    board = state.board
    moves: list[Move] = []

    for name in sorted(vehicles):
        v = vehicles[name]
        if v.orientation == "H":
            row = v.cells[0] // BOARD_SIZE
            left_col = v.cells[0] % BOARD_SIZE
            right_col = v.cells[-1] % BOARD_SIZE
            # slide left while cells stay empty
            steps = 0
            for col in range(left_col - 1, -1, -1):
                if board[row * BOARD_SIZE + col] != EMPTY:
                    break
                steps += 1
                moves.append(Move(name, "L", steps))
            # slide right while cells stay empty
            steps = 0
            for col in range(right_col + 1, BOARD_SIZE):
                if board[row * BOARD_SIZE + col] != EMPTY:
                    break
                steps += 1
                moves.append(Move(name, "R", steps))
        else:
            col = v.cells[0] % BOARD_SIZE
            top_row = v.cells[0] // BOARD_SIZE
            bottom_row = v.cells[-1] // BOARD_SIZE
            # slide up
            steps = 0
            for row in range(top_row - 1, -1, -1):
                if board[row * BOARD_SIZE + col] != EMPTY:
                    break
                steps += 1
                moves.append(Move(name, "U", steps))
            # slide down
            steps = 0
            for row in range(bottom_row + 1, BOARD_SIZE):
                if board[row * BOARD_SIZE + col] != EMPTY:
                    break
                steps += 1
                moves.append(Move(name, "D", steps))
    return moves


def apply_move(state: RushHourState, move: Move) -> RushHourState:
    """Return a new state with the move applied. Raises ValueError for illegal moves."""
    vehicles = find_vehicles(state)
    if move.car not in vehicles:
        raise ValueError(f"no vehicle named '{move.car}' on the board")
    if move.steps < 1:
        raise ValueError(f"move steps must be at least 1, got {move.steps}")
    if move.direction not in _DELTAS:
        raise ValueError(f"unknown direction '{move.direction}'")

    v = vehicles[move.car]
    if v.orientation == "H" and move.direction not in ("L", "R"):
        raise ValueError(f"vehicle '{move.car}' is horizontal and can only move L or R")
    if v.orientation == "V" and move.direction not in ("U", "D"):
        raise ValueError(f"vehicle '{move.car}' is vertical and can only move U or D")

    unit = _DELTAS[move.direction]
    # temp board without the moving vehicle, so it does not collide with itself
    temp = [EMPTY if ch == move.car else ch for ch in state.board]

    for step in range(1, move.steps + 1):
        for cell in v.cells:
            new_cell = cell + unit * step
            if new_cell < 0 or new_cell >= BOARD_CELLS:
                raise ValueError(f"move {move} leaves the board")
            if move.direction in ("L", "R") and new_cell // BOARD_SIZE != cell // BOARD_SIZE:
                raise ValueError(f"move {move} leaves the board")
            if temp[new_cell] != EMPTY:
                raise ValueError(f"move {move} collides with vehicle '{temp[new_cell]}'")

    for cell in v.cells:
        temp[cell + unit * move.steps] = move.car
    return RushHourState(board="".join(temp))
