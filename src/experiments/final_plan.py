"""Final experiment plan: load it, validate it, and turn it into commands.

The functions here only BUILD command strings for the final runs — nothing
is executed. The generated command file goes under results/ and is not
committed.
"""

import json
from pathlib import Path

from src.experiments.official_benchmarks import (
    OFFICIAL_CVRP_INSTANCES,
    expected_instance_path,
    load_bks_table,
)

REQUIRED_SIX = ["sa", "tabu", "aco", "ga_island", "alns", "bnb_lds"]


def load_final_plan(path="configs/final_experiment_plan.json") -> dict:
    with open(path) as f:
        return json.load(f)


def validate_final_plan(plan: dict, require_official_data: bool = False):
    """Return (ok, errors). Normal validation failures never raise."""
    errors = []
    cvrp = plan.get("cvrp", {})
    ackley = plan.get("ackley", {})
    rushhour = plan.get("rushhour", {})

    if cvrp.get("instances") != OFFICIAL_CVRP_INSTANCES:
        errors.append("cvrp instances must be exactly the 6 official instances in order")
    if cvrp.get("required_algorithms") != REQUIRED_SIX:
        errors.append("cvrp required algorithms must be exactly: " + ", ".join(REQUIRED_SIX))
    if ackley.get("dimension") != 10:
        errors.append("ackley dimension must be 10")
    if ackley.get("required_algorithms") != REQUIRED_SIX:
        errors.append("ackley required algorithms must be exactly: " + ", ".join(REQUIRED_SIX))

    for part_name, part in (("cvrp", cvrp), ("ackley", ackley), ("rushhour", rushhour)):
        if not part.get("seeds"):
            errors.append(f"{part_name} seeds must not be empty")

    per_instance = cvrp.get("per_instance", {})
    for name in OFFICIAL_CVRP_INSTANCES:
        settings = per_instance.get(name)
        if settings is None:
            errors.append(f"cvrp per_instance settings missing for {name}")
        elif settings.get("budget", 0) <= 0 or settings.get("timeout_sec", 0) <= 0:
            errors.append(f"cvrp budget/timeout for {name} must be positive")
    if ackley.get("budget", 0) <= 0 or ackley.get("timeout_sec", 0) <= 0:
        errors.append("ackley budget and timeout must be positive")

    try:
        load_bks_table(cvrp.get("bks_path", "data/cvrp_bks.csv"))
    except (OSError, ValueError, KeyError) as exc:
        errors.append(f"BKS table problem: {exc}")

    if require_official_data:
        data_dir = cvrp.get("data_dir", "data/official_cvrp")
        for name in OFFICIAL_CVRP_INSTANCES:
            path = expected_instance_path(name, data_dir)
            if not path.exists():
                errors.append(f"missing official file: {path}")

    return (len(errors) == 0, errors)


# ---------- command builders (build only, never run) ----------

def build_cvrp_commands(plan: dict) -> list[str]:
    """One command per official instance, with its own budget and timeout."""
    cvrp = plan["cvrp"]
    output_dir = plan.get("output_dir", "results/final_experiments")
    seeds = " ".join(str(s) for s in cvrp["seeds"])
    algorithms = " ".join(cvrp["required_algorithms"])
    commands = []
    for name in cvrp["instances"]:
        settings = cvrp["per_instance"][name]
        instance_path = expected_instance_path(name, cvrp["data_dir"])
        commands.append(
            "python scripts/run_experiments.py --part cvrp"
            f" --instances {instance_path}"
            f" --algorithms {algorithms} --include-baseline"
            f" --seeds {seeds}"
            f" --budget {settings['budget']}"
            f" --timeout {settings['timeout_sec']}"
            f" --bks {cvrp['bks_path']}"
            f" --output {output_dir}/raw/cvrp_{name}.csv"
        )
    return commands


def build_ackley_commands(plan: dict) -> list[str]:
    ackley = plan["ackley"]
    output_dir = plan.get("output_dir", "results/final_experiments")
    seeds = " ".join(str(s) for s in ackley["seeds"])
    algorithms = " ".join(ackley["required_algorithms"])
    return [
        "python scripts/run_experiments.py --part ackley"
        f" --algorithms {algorithms} --include-random-search"
        f" --seeds {seeds}"
        f" --budget {ackley['budget']}"
        f" --timeout {ackley['timeout_sec']}"
        f" --dimension {ackley['dimension']}"
        f" --lower {ackley['lower']}"
        f" --upper {ackley['upper']}"
        f" --output {output_dir}/raw/ackley_d{ackley['dimension']}.csv"
    ]


def build_rushhour_commands(plan: dict) -> list[str]:
    rh = plan["rushhour"]
    output_dir = plan.get("output_dir", "results/final_experiments")
    seeds = " ".join(str(s) for s in rh["seeds"])
    return [
        "python scripts/compare_gp_gep_rushhour.py"
        f" --train-puzzles {rh['train_puzzles']}"
        f" --eval-puzzles {rh['eval_puzzles']}"
        f" --seeds {seeds}"
        f" --generations {rh['generations']}"
        f" --population-size {rh['population_size']}"
        f" --gp-max-depth {rh['gp_max_depth']}"
        f" --gep-head-length {rh['gep_head_length']}"
        f" --crossover-rate {rh['crossover_rate']}"
        f" --gp-mutation-rate {rh['gp_mutation_rate']}"
        f" --gep-mutation-rate {rh['gep_mutation_rate']}"
        f" --max-nodes-per-puzzle {rh['max_nodes_per_puzzle']}"
        f" --max-time-per-puzzle {rh['max_time_per_puzzle']}"
        f" --max-total-time {rh['max_total_time']}"
        f" --output {output_dir}/raw/gp_gep_comparison_runs.csv"
        f" --summary-output {output_dir}/raw/gp_gep_comparison_summary.txt"
    ]


def build_asset_commands(plan: dict) -> list[str]:
    """Summaries and report assets, to run after the raw runs exist."""
    cvrp = plan["cvrp"]
    ackley = plan["ackley"]
    output_dir = plan.get("output_dir", "results/final_experiments")
    assets_dir = plan.get("assets", {}).get("output_dir",
                                            f"{output_dir}/report_assets")
    commands = []
    for name in cvrp["instances"]:
        commands.append(
            "python scripts/summarize_experiments.py --part cvrp"
            f" --input {output_dir}/raw/cvrp_{name}.csv"
            f" --output {output_dir}/summary/cvrp_{name}_summary.csv"
        )
    commands.append(
        "python scripts/summarize_experiments.py --part ackley"
        f" --input {output_dir}/raw/ackley_d{ackley['dimension']}.csv"
        f" --output {output_dir}/summary/ackley_d{ackley['dimension']}_summary.csv"
    )
    # note: cvrp_all_instances.csv is the combined CSV that the aggregation
    # step of the next stage produces from the per-instance raw files
    commands.append(
        "# combined CVRP file below is produced by the next-stage aggregation step"
    )
    commands.append(
        "python scripts/generate_report_assets.py"
        f" --cvrp-raw {output_dir}/raw/cvrp_all_instances.csv"
        f" --cvrp-summary {output_dir}/summary/cvrp_all_instances_summary.csv"
        f" --ackley-raw {output_dir}/raw/ackley_d{ackley['dimension']}.csv"
        f" --ackley-summary {output_dir}/summary/ackley_d{ackley['dimension']}_summary.csv"
        f" --gp-gep-csv {output_dir}/raw/gp_gep_comparison_runs.csv"
        f" --gp-gep-summary {output_dir}/raw/gp_gep_comparison_summary.txt"
        f" --output-dir {assets_dir}"
    )
    return commands


def build_all_commands(plan: dict) -> dict:
    return {
        "cvrp": build_cvrp_commands(plan),
        "ackley": build_ackley_commands(plan),
        "rushhour": build_rushhour_commands(plan),
        "assets": build_asset_commands(plan),
    }


def write_command_file(commands: dict, path) -> Path:
    """Plain text command file with one section per part."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for section, section_commands in commands.items():
        lines.append(f"# ---- {section} ----")
        lines.extend(section_commands)
        lines.append("")
    path.write_text("\n".join(lines))
    return path
