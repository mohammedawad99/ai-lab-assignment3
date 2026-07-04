"""Summaries of raw experiment CSVs into compact per-algorithm tables."""

import csv
import math
from pathlib import Path

CVRP_SUMMARY_FIELDNAMES = [
    "part", "instance", "algorithm", "runs", "feasible_runs",
    "best_cost", "mean_cost", "std_cost",
    "best_gap_percent", "mean_gap_percent",
    "mean_elapsed_time", "mean_cpu_time", "best_seed",
]

ACKLEY_SUMMARY_FIELDNAMES = [
    "part", "algorithm", "runs",
    "best_value", "mean_best_value", "std_best_value",
    "mean_distance_from_origin", "mean_elapsed_time", "mean_cpu_time",
    "best_seed",
]


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def mean(values):
    if not values:
        return None
    return sum(values) / len(values)


def std(values):
    """Sample standard deviation; 0.0 for a single value, None for none."""
    if not values:
        return None
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def _fmt(value, decimals=4):
    return "" if value is None else f"{value:.{decimals}f}"


def _is_true_like(value):
    return str(value).strip().lower() in ("true", "yes", "1")


def _floats(rows, field):
    return [v for v in (safe_float(row.get(field)) for row in rows) if v is not None]


def summarize_cvrp_rows(rows: list[dict]) -> list[dict]:
    """One summary row per (instance, algorithm)."""
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (row.get("instance", ""), row.get("algorithm", ""))
        groups.setdefault(key, []).append(row)

    summary = []
    for (instance, algorithm), group in sorted(groups.items()):
        costed = [(safe_float(row.get("best_cost")), row) for row in group]
        costed = [(cost, row) for cost, row in costed if cost is not None]
        costs = [cost for cost, _ in costed]
        gaps = _floats(group, "gap_percent")
        best_seed = ""
        if costed:
            best_seed = min(costed, key=lambda pair: pair[0])[1].get("seed", "")
        summary.append({
            "part": "cvrp",
            "instance": instance,
            "algorithm": algorithm,
            "runs": len(group),
            "feasible_runs": sum(1 for row in group if _is_true_like(row.get("feasible"))),
            "best_cost": _fmt(min(costs) if costs else None),
            "mean_cost": _fmt(mean(costs)),
            "std_cost": _fmt(std(costs)),
            "best_gap_percent": _fmt(min(gaps) if gaps else None),
            "mean_gap_percent": _fmt(mean(gaps)),
            "mean_elapsed_time": _fmt(mean(_floats(group, "elapsed_time")), 6),
            "mean_cpu_time": _fmt(mean(_floats(group, "cpu_time")), 6),
            "best_seed": best_seed,
        })
    return summary


def summarize_ackley_rows(rows: list[dict]) -> list[dict]:
    """One summary row per algorithm."""
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row.get("algorithm", ""), []).append(row)

    summary = []
    for algorithm, group in sorted(groups.items()):
        valued = [(safe_float(row.get("best_value")), row) for row in group]
        valued = [(value, row) for value, row in valued if value is not None]
        values = [value for value, _ in valued]
        best_seed = ""
        if valued:
            best_seed = min(valued, key=lambda pair: pair[0])[1].get("seed", "")
        summary.append({
            "part": "ackley",
            "algorithm": algorithm,
            "runs": len(group),
            "best_value": _fmt(min(values) if values else None, 6),
            "mean_best_value": _fmt(mean(values), 6),
            "std_best_value": _fmt(std(values), 6),
            "mean_distance_from_origin": _fmt(mean(_floats(group, "distance_from_origin")), 6),
            "mean_elapsed_time": _fmt(mean(_floats(group, "elapsed_time")), 6),
            "mean_cpu_time": _fmt(mean(_floats(group, "cpu_time")), 6),
            "best_seed": best_seed,
        })
    return summary


def read_csv_rows(path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_summary_csv(path, rows: list[dict], fieldnames: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
