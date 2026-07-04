"""Execute the final experiment plan and aggregate the results.

Everything runs through the existing Python functions (no shell calls). The
suite is resumable: raw CSVs that already exist with rows are not rerun, so
an interrupted long CVRP run does not force a full restart. All output goes
under the plan's output_dir (results/final_experiments), which is gitignored.
"""

import csv
import json
import time
from pathlib import Path

from src.experiments.ackley_experiments import ACKLEY_FIELDNAMES, run_ackley_experiments
from src.experiments.csv_utils import write_dict_rows
from src.experiments.cvrp_experiments import CVRP_FIELDNAMES, run_cvrp_experiments
from src.experiments.final_plan import load_final_plan, validate_final_plan
from src.experiments.official_benchmarks import expected_instance_path, load_bks_table
from src.experiments.report_assets import generate_report_assets
from src.experiments.summary import (
    ACKLEY_SUMMARY_FIELDNAMES,
    CVRP_SUMMARY_FIELDNAMES,
    read_csv_rows,
    summarize_ackley_rows,
    summarize_cvrp_rows,
    write_summary_csv,
)
from src.rushhour.gp_gep_comparison import (
    compare_gp_gep,
    write_comparison_summary_txt,
    write_run_summaries_csv,
)
from src.rushhour.puzzle_sets import load_puzzle_set


def ensure_dir(path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def combine_csv_files(input_paths, output_path) -> dict:
    """Concatenate CSVs with the same columns into one file with one header.
    Missing inputs are skipped and reported."""
    fieldnames = None
    rows = []
    missing = []
    used = 0
    for path in input_paths:
        path = Path(path)
        if not path.exists():
            missing.append(str(path))
            continue
        used += 1
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            rows.extend(reader)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames or [])
        writer.writeheader()
        writer.writerows(rows)

    return {
        "output_path": str(output_path),
        "input_count": used,
        "missing_count": len(missing),
        "missing_paths": missing,
        "row_count": len(rows),
    }


def write_execution_manifest(path, manifest: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    return path


# ---------- part runners ----------

def run_final_cvrp(plan: dict, output_dir, resume: bool = True) -> list[dict]:
    cvrp = plan["cvrp"]
    bks = load_bks_table(cvrp["bks_path"])
    raw_dir = Path(output_dir) / "raw"
    info = []
    for name in cvrp["instances"]:
        settings = cvrp["per_instance"][name]
        output_csv = raw_dir / f"cvrp_{name}.csv"
        entry = {
            "instance": name,
            "output_csv": str(output_csv),
            "rows": 0,
            "skipped_existing": False,
            "budget": settings["budget"],
            "timeout_sec": settings["timeout_sec"],
            "elapsed_time": 0.0,
        }
        if resume and output_csv.exists():
            existing = read_csv_rows(output_csv)
            if existing:
                entry["rows"] = len(existing)
                entry["skipped_existing"] = True
                info.append(entry)
                continue

        start = time.perf_counter()
        rows = run_cvrp_experiments(
            [expected_instance_path(name, cvrp["data_dir"])],
            cvrp["algorithms"], cvrp["seeds"],
            budget=settings["budget"],
            timeout_sec=settings["timeout_sec"],
            output_dir=Path(output_dir) / "cvrp_solutions" / name,
            bks_by_instance=bks,
        )
        write_dict_rows(output_csv, rows, CVRP_FIELDNAMES)
        entry["rows"] = len(rows)
        entry["elapsed_time"] = time.perf_counter() - start
        info.append(entry)
    return info


def run_final_ackley(plan: dict, output_dir, resume: bool = True) -> dict:
    ackley = plan["ackley"]
    output_csv = Path(output_dir) / "raw" / f"ackley_d{ackley['dimension']}.csv"
    entry = {
        "output_csv": str(output_csv),
        "rows": 0,
        "skipped_existing": False,
        "elapsed_time": 0.0,
    }
    if resume and output_csv.exists():
        existing = read_csv_rows(output_csv)
        if existing:
            entry["rows"] = len(existing)
            entry["skipped_existing"] = True
            return entry

    start = time.perf_counter()
    rows = run_ackley_experiments(
        ackley["algorithms"], ackley["seeds"],
        budget=ackley["budget"], timeout_sec=ackley["timeout_sec"],
        dimension=ackley["dimension"], lower=ackley["lower"], upper=ackley["upper"],
    )
    write_dict_rows(output_csv, rows, ACKLEY_FIELDNAMES)
    entry["rows"] = len(rows)
    entry["elapsed_time"] = time.perf_counter() - start
    return entry


def run_final_rushhour(plan: dict, output_dir, resume: bool = True) -> dict:
    rh = plan["rushhour"]
    raw_dir = Path(output_dir) / "raw"
    csv_path = raw_dir / "gp_gep_comparison_runs.csv"
    txt_path = raw_dir / "gp_gep_comparison_summary.txt"
    entry = {
        "output_csv": str(csv_path),
        "summary_txt": str(txt_path),
        "rows": 0,
        "skipped_existing": False,
        "elapsed_time": 0.0,
    }
    if resume and csv_path.exists() and txt_path.exists():
        existing = read_csv_rows(csv_path)
        if existing:
            entry["rows"] = len(existing)
            entry["skipped_existing"] = True
            return entry

    start = time.perf_counter()
    train = load_puzzle_set(rh["train_puzzles"])
    eval_puzzles = load_puzzle_set(rh["eval_puzzles"])
    summary = compare_gp_gep(
        train, eval_puzzles, rh["seeds"],
        generations=rh["generations"],
        population_size=rh["population_size"],
        gp_max_depth=rh["gp_max_depth"],
        gep_head_length=rh["gep_head_length"],
        crossover_rate=rh["crossover_rate"],
        gp_mutation_rate=rh["gp_mutation_rate"],
        gep_mutation_rate=rh["gep_mutation_rate"],
        max_nodes_per_puzzle=rh["max_nodes_per_puzzle"],
        max_time_per_puzzle_sec=rh["max_time_per_puzzle"],
        max_total_time_sec=rh["max_total_time"],
    )
    write_run_summaries_csv(summary, csv_path)
    write_comparison_summary_txt(summary, txt_path)
    entry["rows"] = len(summary.gp_runs) + len(summary.gep_runs)
    entry["elapsed_time"] = time.perf_counter() - start
    return entry


# ---------- aggregation and assets ----------

def aggregate_final_results(plan: dict, output_dir) -> dict:
    cvrp = plan["cvrp"]
    ackley = plan["ackley"]
    raw_dir = Path(output_dir) / "raw"
    summary_dir = Path(output_dir) / "summary"

    per_instance_paths = [raw_dir / f"cvrp_{name}.csv" for name in cvrp["instances"]]
    combined_path = raw_dir / "cvrp_all_instances.csv"
    combined = combine_csv_files(per_instance_paths, combined_path)

    info = {"cvrp_combined": combined, "summaries": [], "missing": []}

    all_rows = read_csv_rows(combined_path)
    all_summary_path = summary_dir / "cvrp_all_summary.csv"
    write_summary_csv(all_summary_path, summarize_cvrp_rows(all_rows),
                      CVRP_SUMMARY_FIELDNAMES)
    info["summaries"].append(str(all_summary_path))

    for name in cvrp["instances"]:
        raw_path = raw_dir / f"cvrp_{name}.csv"
        if not raw_path.exists():
            info["missing"].append(str(raw_path))
            continue
        summary_path = summary_dir / f"cvrp_{name}_summary.csv"
        write_summary_csv(summary_path, summarize_cvrp_rows(read_csv_rows(raw_path)),
                          CVRP_SUMMARY_FIELDNAMES)
        info["summaries"].append(str(summary_path))

    ackley_raw = raw_dir / f"ackley_d{ackley['dimension']}.csv"
    if ackley_raw.exists():
        ackley_summary = summary_dir / f"ackley_d{ackley['dimension']}_summary.csv"
        write_summary_csv(ackley_summary,
                          summarize_ackley_rows(read_csv_rows(ackley_raw)),
                          ACKLEY_SUMMARY_FIELDNAMES)
        info["summaries"].append(str(ackley_summary))
    else:
        info["missing"].append(str(ackley_raw))

    return info


def generate_final_assets(plan: dict, output_dir) -> list[str]:
    output_dir = Path(output_dir)
    ackley_dim = plan["ackley"]["dimension"]
    assets_dir = plan.get("assets", {}).get("output_dir",
                                            str(output_dir / "report_assets"))
    created = generate_report_assets(
        cvrp_raw=output_dir / "raw" / "cvrp_all_instances.csv",
        cvrp_summary=output_dir / "summary" / "cvrp_all_summary.csv",
        ackley_raw=output_dir / "raw" / f"ackley_d{ackley_dim}.csv",
        ackley_summary=output_dir / "summary" / f"ackley_d{ackley_dim}_summary.csv",
        gp_gep_csv=output_dir / "raw" / "gp_gep_comparison_runs.csv",
        gp_gep_summary=output_dir / "raw" / "gp_gep_comparison_summary.txt",
        output_dir=assets_dir,
    )
    return [str(path) for path in created]


# ---------- full suite ----------

def run_final_experiment_suite(plan_path="configs/final_experiment_plan.json",
                               resume: bool = True, run_cvrp: bool = True,
                               run_ackley: bool = True,
                               run_rushhour: bool = True) -> dict:
    plan = load_final_plan(plan_path)
    ok, errors = validate_final_plan(plan, require_official_data=True)
    if not ok:
        raise ValueError("final plan validation failed: " + " | ".join(errors))

    output_dir = plan.get("output_dir", "results/final_experiments")
    manifest = {
        "plan_path": str(plan_path),
        "resume": resume,
        "run_cvrp": run_cvrp,
        "run_ackley": run_ackley,
        "run_rushhour": run_rushhour,
        "cvrp_run_info": [],
        "ackley_run_info": {},
        "rushhour_run_info": {},
    }

    if run_cvrp:
        manifest["cvrp_run_info"] = run_final_cvrp(plan, output_dir, resume=resume)
    if run_ackley:
        manifest["ackley_run_info"] = run_final_ackley(plan, output_dir, resume=resume)
    if run_rushhour:
        manifest["rushhour_run_info"] = run_final_rushhour(plan, output_dir, resume=resume)

    manifest["aggregation_info"] = aggregate_final_results(plan, output_dir)
    manifest["asset_paths"] = generate_final_assets(plan, output_dir)

    manifest_path = Path(output_dir) / "final_execution_manifest.json"
    write_execution_manifest(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest
