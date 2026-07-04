"""Small CSV helpers for experiment results."""

import csv
from pathlib import Path


def write_dict_rows(path, rows: list[dict], fieldnames: list[str]) -> None:
    """Write header + rows, overwriting the file. Header is written even if
    rows is empty."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_dict_rows(path, rows: list[dict], fieldnames: list[str]) -> None:
    """Append rows; the header is only written when the file is new."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new_file = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new_file:
            writer.writeheader()
        writer.writerows(rows)
