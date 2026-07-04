"""Parsing Rush Hour board strings and detecting vehicles."""

from src.rushhour.model import BOARD_CELLS, BOARD_SIZE, EMPTY, RED_CAR, RushHourState, Vehicle


def parse_board_string(text: str) -> RushHourState:
    """Build a state from a raw string, ignoring whitespace and newlines."""
    board = "".join(text.split())
    if len(board) != BOARD_CELLS:
        raise ValueError(f"board must have {BOARD_CELLS} cells, got {len(board)}")
    if RED_CAR not in board:
        raise ValueError("board has no red car 'X'")
    return RushHourState(board=board)


def board_to_rows(state: RushHourState) -> list[str]:
    """Return the board as 6 row strings, useful for printing and debugging."""
    return [state.board[r * BOARD_SIZE:(r + 1) * BOARD_SIZE] for r in range(BOARD_SIZE)]


def find_vehicles(state: RushHourState) -> dict[str, Vehicle]:
    """Detect all vehicles and validate that each one is a straight 2- or 3-cell line."""
    positions: dict[str, list[int]] = {}
    for idx, ch in enumerate(state.board):
        if ch != EMPTY:
            positions.setdefault(ch, []).append(idx)

    vehicles: dict[str, Vehicle] = {}
    for name in sorted(positions):
        cells = tuple(sorted(positions[name]))
        length = len(cells)
        if length not in (2, 3):
            raise ValueError(f"vehicle '{name}' has length {length}, expected 2 or 3")

        rows = {c // BOARD_SIZE for c in cells}
        cols = {c % BOARD_SIZE for c in cells}
        horizontal = len(rows) == 1 and all(
            cells[i + 1] - cells[i] == 1 for i in range(length - 1)
        )
        vertical = len(cols) == 1 and all(
            cells[i + 1] - cells[i] == BOARD_SIZE for i in range(length - 1)
        )
        if horizontal:
            orientation = "H"
        elif vertical:
            orientation = "V"
        else:
            raise ValueError(f"vehicle '{name}' cells are not a straight contiguous line")

        vehicles[name] = Vehicle(name=name, cells=cells, orientation=orientation, length=length)
    return vehicles
