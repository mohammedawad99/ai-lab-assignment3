"""First small run on the official CVRP data, if the files are placed.

This never downloads or invents data: it checks which official .vrp files
are actually present, runs a small mini-experiment on them, and can skip
cleanly (marked as skipped) when the data is not placed yet.
"""

from pathlib import Path

from src.experiments.csv_utils import write_dict_rows
from src.experiments.cvrp_experiments import CVRP_FIELDNAMES, run_cvrp_experiments
from src.experiments.official_benchmarks import (
    DEFAULT_BKS_PATH,
    DEFAULT_OFFICIAL_CVRP_DIR,
    OFFICIAL_CVRP_INSTANCES,
    check_official_data,
    expected_instance_path,
    load_bks_table,
)
from src.experiments.summary import (
    CVRP_SUMMARY_FIELDNAMES,
    summarize_cvrp_rows,
    write_summary_csv,
)

OFFICIAL_MINI_DEFAULT_ALGORITHMS = [
    "baseline", "sa", "tabu", "aco", "ga_island", "alns", "bnb_lds",
]
OFFICIAL_MINI_DEFAULT_SEEDS = [42]
OFFICIAL_MINI_DEFAULT_BUDGET = 20
OFFICIAL_MINI_DEFAULT_TIMEOUT_SEC = 30.0
OFFICIAL_MINI_DEFAULT_INSTANCE = "P-n16-k8"


def available_official_instances(data_dir=DEFAULT_OFFICIAL_CVRP_DIR,
                                 bks_path=DEFAULT_BKS_PATH) -> list[str]:
    """Official instance names whose .vrp file is actually present."""
    report = check_official_data(data_dir=data_dir, bks_path=bks_path, strict=False)
    if report["unexpected_files"]:
        raise ValueError(
            f"unexpected .vrp files in {data_dir}: {', '.join(report['unexpected_files'])}"
        )
    if report["errors"]:
        raise ValueError("official data problems: " + " | ".join(report["errors"]))
    return report["found_instances"]


def choose_official_mini_instances(requested_instances=None,
                                   data_dir=DEFAULT_OFFICIAL_CVRP_DIR,
                                   bks_path=DEFAULT_BKS_PATH):
    """Pick the instances for the mini-run. Returns (selected, missing)."""
    available = set(available_official_instances(data_dir, bks_path))

    if requested_instances is not None:
        for name in requested_instances:
            if name not in OFFICIAL_CVRP_INSTANCES:
                raise ValueError(f"'{name}' is not one of the six official instances")
        selected = [name for name in requested_instances if name in available]
        missing = [name for name in requested_instances if name not in available]
        return selected, missing

    # default: the smallest instance first, otherwise the first one present
    if OFFICIAL_MINI_DEFAULT_INSTANCE in available:
        return [OFFICIAL_MINI_DEFAULT_INSTANCE], []
    for name in OFFICIAL_CVRP_INSTANCES:
        if name in available:
            return [name], []
    return [], [OFFICIAL_MINI_DEFAULT_INSTANCE]


def run_official_mini_experiment(data_dir=DEFAULT_OFFICIAL_CVRP_DIR,
                                 bks_path=DEFAULT_BKS_PATH,
                                 requested_instances=None, algorithms=None,
                                 seeds=None,
                                 budget=OFFICIAL_MINI_DEFAULT_BUDGET,
                                 timeout_sec=OFFICIAL_MINI_DEFAULT_TIMEOUT_SEC,
                                 output_dir="results/official_mini",
                                 include_unused_vehicles=False,
                                 allow_missing_data=False) -> dict:
    bks = load_bks_table(bks_path)
    selected, missing = choose_official_mini_instances(requested_instances,
                                                       data_dir, bks_path)
    if algorithms is None:
        algorithms = list(OFFICIAL_MINI_DEFAULT_ALGORITHMS)
    if seeds is None:
        seeds = list(OFFICIAL_MINI_DEFAULT_SEEDS)

    if not selected:
        if allow_missing_data:
            return {
                "ok": True,
                "skipped": True,
                "reason": "missing official data",
                "selected_instances": [],
                "missing_instances": missing,
                "raw_csv": "",
                "summary_csv": "",
                "solution_dir": "",
                "raw_rows": [],
                "summary_rows": [],
            }
        raise FileNotFoundError(
            f"official .vrp files are missing under '{data_dir}'; place them "
            f"there first, or run with allow_missing_data=True to skip cleanly"
        )

    output_dir = Path(output_dir)
    instance_paths = [expected_instance_path(name, data_dir) for name in selected]
    raw_rows = run_cvrp_experiments(
        instance_paths, algorithms, seeds, budget=budget, timeout_sec=timeout_sec,
        output_dir=output_dir, include_unused_vehicles=include_unused_vehicles,
        bks_by_instance=bks,
    )

    raw_csv = output_dir / "raw" / "official_mini_cvrp_results.csv"
    write_dict_rows(raw_csv, raw_rows, CVRP_FIELDNAMES)
    summary_rows = summarize_cvrp_rows(raw_rows)
    summary_csv = output_dir / "summary" / "official_mini_cvrp_summary.csv"
    write_summary_csv(summary_csv, summary_rows, CVRP_SUMMARY_FIELDNAMES)

    return {
        "ok": True,
        "skipped": False,
        "reason": "",
        "selected_instances": selected,
        "missing_instances": missing,
        "raw_csv": str(raw_csv),
        "summary_csv": str(summary_csv),
        "solution_dir": str(output_dir / "cvrp_solutions"),
        "raw_rows": raw_rows,
        "summary_rows": summary_rows,
    }
