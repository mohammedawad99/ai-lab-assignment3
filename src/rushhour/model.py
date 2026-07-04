"""Data classes for Rush Hour boards, moves and search results."""

from dataclasses import dataclass

BOARD_SIZE = 6
BOARD_CELLS = BOARD_SIZE * BOARD_SIZE
RED_CAR = "X"
EMPTY = "."


@dataclass(frozen=True)
class Vehicle:
    name: str
    cells: tuple[int, ...]  # board indexes (row * 6 + col), sorted
    orientation: str  # "H" or "V"
    length: int


@dataclass(frozen=True)
class Move:
    car: str
    direction: str  # "U", "D", "L", "R"
    steps: int

    def __str__(self):
        # course-like token, e.g. Move("A", "D", 1) -> "1AD"
        return f"{self.steps}{self.car}{self.direction}"


@dataclass(frozen=True)
class RushHourState:
    board: str  # 36 characters, "." is empty, "X" is the red car

    def __post_init__(self):
        if len(self.board) != BOARD_CELLS:
            raise ValueError(
                f"board string must have {BOARD_CELLS} characters, got {len(self.board)}"
            )


@dataclass
class SearchResult:
    solved: bool
    moves: list[Move]
    cost: int
    expanded_nodes: int
    generated_nodes: int
    elapsed_time: float
    cpu_time: float
    stopped_reason: str | None = None  # "solved", "max_nodes", "timeout", "no_solution"
