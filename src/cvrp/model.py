"""Data classes for CVRP instances and solutions."""

from dataclasses import dataclass


@dataclass
class CVRPInstance:
    """A CVRP instance with internal node ids 0..n, where 0 is the depot."""

    name: str
    capacity: float
    vehicle_count: int
    coordinates: dict[int, tuple[float, float]]
    demands: dict[int, float]
    depot_id: int = 0
    # original_ids[internal_id] = node id in the source file (CVRPLIB ids start at 1)
    original_ids: dict[int, int] | None = None

    @property
    def node_count(self) -> int:
        return len(self.coordinates)

    @property
    def customer_ids(self) -> list[int]:
        return [i for i in self.coordinates if i != self.depot_id]


@dataclass
class CVRPSolution:
    """A list of routes. Each route starts and ends at the depot, e.g. [0, 1, 2, 0]."""

    routes: list[list[int]]
    cost: float | None = None
