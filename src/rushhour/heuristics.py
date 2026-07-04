"""Baseline heuristics for A*. Simple baselines, not claimed to be admissible."""

from src.rushhour.features import blocking_cars, red_car_distance_to_exit
from src.rushhour.model import RushHourState


def zero_heuristic(state: RushHourState) -> int:
    return 0


def blocking_heuristic(state: RushHourState) -> int:
    return blocking_cars(state)


def blocking_distance_heuristic(state: RushHourState) -> int:
    return blocking_cars(state) + red_car_distance_to_exit(state)


# names used by the CLI --heuristic flag
BASELINE_HEURISTICS = {
    "zero": zero_heuristic,
    "blocking": blocking_heuristic,
    "blocking_distance": blocking_distance_heuristic,
}
