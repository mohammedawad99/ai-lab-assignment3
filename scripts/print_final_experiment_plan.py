"""Validate the final experiment plan and print the exact run commands.

Usage:
    python scripts/print_final_experiment_plan.py
    python scripts/print_final_experiment_plan.py --write-commands results/final_experiment_commands.txt
    python scripts/print_final_experiment_plan.py --require-official-data
"""

import argparse
import sys
from pathlib import Path

# make "src" importable when the script is run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments.final_plan import (
    build_all_commands,
    load_final_plan,
    validate_final_plan,
    write_command_file,
)


def main():
    parser = argparse.ArgumentParser(description="Print the final experiment plan.")
    parser.add_argument("--plan", default="configs/final_experiment_plan.json")
    parser.add_argument("--require-official-data", action="store_true",
                        help="also require all six official .vrp files to exist")
    parser.add_argument("--write-commands",
                        help="optional path for a plain text command file")
    args = parser.parse_args()

    plan = load_final_plan(args.plan)
    ok, errors = validate_final_plan(plan,
                                     require_official_data=args.require_official_data)

    print(f"plan: {args.plan}")
    print(f"validation: {'ok' if ok else 'FAILED'}")
    for error in errors:
        print(f"  error: {error}")

    cvrp = plan.get("cvrp", {})
    ackley = plan.get("ackley", {})
    rushhour = plan.get("rushhour", {})
    print(f"cvrp instances: {', '.join(cvrp.get('instances', []))}")
    print(f"cvrp seeds: {' '.join(str(s) for s in cvrp.get('seeds', []))}")
    print(f"ackley dimension: {ackley.get('dimension')}")
    print(f"ackley seeds: {' '.join(str(s) for s in ackley.get('seeds', []))}")
    print(f"rushhour seeds: {' '.join(str(s) for s in rushhour.get('seeds', []))}")
    print(f"output dir: {plan.get('output_dir')}")

    commands = build_all_commands(plan)
    print(f"cvrp commands: {len(commands['cvrp'])}")
    print(f"ackley commands: {len(commands['ackley'])}")
    print(f"rushhour commands: {len(commands['rushhour'])}")
    print(f"asset commands: {len(commands['assets'])}")
    for section, section_commands in commands.items():
        print(f"\n# ---- {section} ----")
        for command in section_commands:
            print(command)

    if args.write_commands:
        path = write_command_file(commands, args.write_commands)
        print(f"\ncommands written to: {path}")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
