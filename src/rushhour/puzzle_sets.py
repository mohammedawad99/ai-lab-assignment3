"""Loading Rush Hour puzzle sets from a text file.

Format: each puzzle is 6 board rows, puzzles are separated by blank lines,
and lines starting with # are comments.
"""

from src.rushhour.model import RushHourState
from src.rushhour.parser import parse_board_string


def load_puzzle_set(path) -> list[RushHourState]:
    puzzles: list[RushHourState] = []
    current_rows: list[str] = []

    with open(path) as f:
        lines = f.readlines()

    # extra blank line at the end flushes the last puzzle
    for raw in lines + [""]:
        line = raw.strip()
        if line.startswith("#"):
            continue
        if not line:
            if current_rows:
                puzzles.append(parse_board_string("".join(current_rows)))
                current_rows = []
            continue
        current_rows.append(line)

    return puzzles
