"""Parser for CVRPLIB-style instance files (EUC_2D only).

File node ids usually start at 1. Internally we remap the depot to 0 and the
customers to 1..n. The mapping back to file ids is kept in original_ids.
"""

import re

from src.cvrp.model import CVRPInstance


def parse_cvrplib(path) -> CVRPInstance:
    header: dict[str, str] = {}
    file_coords: dict[int, tuple[float, float]] = {}
    file_demands: dict[int, float] = {}
    depot_file_id: int | None = None
    section = None

    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line == "EOF":
                break
            if line == "NODE_COORD_SECTION":
                section = "coords"
                continue
            if line == "DEMAND_SECTION":
                section = "demands"
                continue
            if line == "DEPOT_SECTION":
                section = "depot"
                continue

            if section == "coords":
                parts = line.split()
                file_coords[int(parts[0])] = (float(parts[1]), float(parts[2]))
            elif section == "demands":
                parts = line.split()
                file_demands[int(parts[0])] = float(parts[1])
            elif section == "depot":
                value = int(float(line.split()[0]))
                if value == -1:
                    section = None
                elif depot_file_id is None:
                    depot_file_id = value
            elif ":" in line:
                key, value = line.split(":", 1)
                header[key.strip().upper()] = value.strip()

    edge_type = header.get("EDGE_WEIGHT_TYPE", "")
    if edge_type != "EUC_2D":
        raise ValueError(f"unsupported EDGE_WEIGHT_TYPE '{edge_type}', only EUC_2D is supported")

    name = header.get("NAME", "")
    capacity = float(header["CAPACITY"])
    dimension = int(header["DIMENSION"])
    if len(file_coords) != dimension:
        raise ValueError(f"DIMENSION is {dimension} but {len(file_coords)} coordinates were read")

    if depot_file_id is None:
        # no DEPOT_SECTION: assume the lowest node id is the depot
        depot_file_id = min(file_coords)

    # vehicle count: explicit VEHICLES field, or the -k<number> suffix in the name
    if "VEHICLES" in header:
        vehicle_count = int(header["VEHICLES"])
    else:
        match = re.search(r"-k(\d+)", name)
        if match is None:
            raise ValueError("cannot determine vehicle count (no VEHICLES field, no -k in NAME)")
        vehicle_count = int(match.group(1))

    # remap: depot -> 0, customers -> 1..n (sorted by file id)
    mapping = {depot_file_id: 0}
    for internal_id, file_id in enumerate(sorted(i for i in file_coords if i != depot_file_id), start=1):
        mapping[file_id] = internal_id

    coordinates = {mapping[i]: file_coords[i] for i in file_coords}
    demands = {mapping[i]: file_demands.get(i, 0.0) for i in file_coords}
    demands[0] = 0.0
    original_ids = {internal: file_id for file_id, internal in mapping.items()}

    return CVRPInstance(
        name=name,
        capacity=capacity,
        vehicle_count=vehicle_count,
        coordinates=coordinates,
        demands=demands,
        depot_id=0,
        original_ids=original_ids,
    )
