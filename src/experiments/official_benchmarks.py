"""Official CVRP benchmark setup: instance manifest, BKS table and readiness check.

The official .vrp files are NOT part of the repository. They must be placed
manually under data/official_cvrp/ with the exact expected filenames. This
module only checks whether everything is ready — it never downloads or
invents benchmark data.
"""

import csv
from pathlib import Path

from src.cvrp.io_cvrplib import parse_cvrplib

OFFICIAL_CVRP_INSTANCES = [
    "P-n16-k8",
    "E-n22-k4",
    "A-n32-k5",
    "A-n80-k10",
    "X-n101-k25",
    "M-n200-k17",
]

DEFAULT_OFFICIAL_CVRP_DIR = "data/official_cvrp"
DEFAULT_BKS_PATH = "data/cvrp_bks.csv"


def load_bks_table(path=DEFAULT_BKS_PATH) -> dict[str, float]:
    """Read the BKS CSV and validate it covers exactly the 6 official instances."""
    table: dict[str, float] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            name = row["instance"].strip()
            if name not in OFFICIAL_CVRP_INSTANCES:
                raise ValueError(f"unknown instance '{name}' in BKS table")
            if name in table:
                raise ValueError(f"duplicate instance '{name}' in BKS table")
            cost = float(row["bks_cost"])
            if cost <= 0:
                raise ValueError(f"BKS for '{name}' must be positive, got {cost}")
            table[name] = cost
    missing = [n for n in OFFICIAL_CVRP_INSTANCES if n not in table]
    if missing:
        raise ValueError(f"BKS table is missing instances: {', '.join(missing)}")
    return table


def expected_instance_path(instance_name: str,
                           data_dir=DEFAULT_OFFICIAL_CVRP_DIR) -> Path:
    return Path(data_dir) / f"{instance_name}.vrp"


def find_official_instance_files(data_dir=DEFAULT_OFFICIAL_CVRP_DIR) -> dict[str, Path]:
    """Map each official instance name to its expected path, if the file exists."""
    found = {}
    for name in OFFICIAL_CVRP_INSTANCES:
        path = expected_instance_path(name, data_dir)
        if path.exists():
            found[name] = path
    return found


def check_official_data(data_dir=DEFAULT_OFFICIAL_CVRP_DIR,
                        bks_path=DEFAULT_BKS_PATH, strict=False) -> dict:
    """Readiness check for the official benchmark setup.

    With strict=False, missing .vrp files are allowed (the data may not be
    placed yet). With strict=True, all 6 files must be present and parse.
    """
    errors: list[str] = []
    bks: dict[str, float] = {}
    try:
        bks = load_bks_table(bks_path)
    except (OSError, ValueError, KeyError) as exc:
        errors.append(f"BKS table problem: {exc}")

    found = find_official_instance_files(data_dir)
    missing = [name for name in OFFICIAL_CVRP_INSTANCES if name not in found]

    unexpected_files = []
    data_dir_path = Path(data_dir)
    if data_dir_path.exists():
        for path in sorted(data_dir_path.glob("*.vrp")):
            if path.stem not in OFFICIAL_CVRP_INSTANCES:
                unexpected_files.append(path.name)

    for name, path in sorted(found.items()):
        try:
            instance = parse_cvrplib(path)
            if not instance.name.lower().startswith(name.lower()):
                errors.append(f"{path.name}: NAME '{instance.name}' does not match {name}")
            if instance.node_count <= 1:
                errors.append(f"{path.name}: node_count must be > 1")
            if instance.capacity <= 0:
                errors.append(f"{path.name}: capacity must be > 0")
            if instance.vehicle_count <= 0:
                errors.append(f"{path.name}: vehicle_count must be > 0")
        except Exception as exc:
            errors.append(f"{path.name}: failed to parse ({exc})")

    ok = not errors and not unexpected_files
    if strict and missing:
        ok = False

    return {
        "ok": ok,
        "data_dir": str(data_dir),
        "bks_path": str(bks_path),
        "expected_instances": list(OFFICIAL_CVRP_INSTANCES),
        "found_instances": sorted(found),
        "missing_instances": missing,
        "unexpected_files": unexpected_files,
        "bks": bks,
        "errors": errors,
    }
