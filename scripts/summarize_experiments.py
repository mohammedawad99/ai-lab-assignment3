"""Summarize a raw experiment CSV into a compact table.

Usage:
    python scripts/summarize_experiments.py --part cvrp \
        --input results/cvrp_smoke.csv --output results/cvrp_smoke_summary.csv
"""

import argparse
import sys
from pathlib import Path

# make "src" importable when the script is run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments.summary import (
    ACKLEY_SUMMARY_FIELDNAMES,
    CVRP_SUMMARY_FIELDNAMES,
    read_csv_rows,
    summarize_ackley_rows,
    summarize_cvrp_rows,
    write_summary_csv,
)


def main():
    parser = argparse.ArgumentParser(description="Summarize raw experiment results.")
    parser.add_argument("--part", required=True, choices=["cvrp", "ackley"])
    parser.add_argument("--input", required=True, help="raw results CSV")
    parser.add_argument("--output", required=True, help="summary CSV to write")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    rows = read_csv_rows(input_path)
    if args.part == "cvrp":
        summary = summarize_cvrp_rows(rows)
        fieldnames = CVRP_SUMMARY_FIELDNAMES
    else:
        summary = summarize_ackley_rows(rows)
        fieldnames = ACKLEY_SUMMARY_FIELDNAMES
    write_summary_csv(args.output, summary, fieldnames)

    print(f"input: {input_path}")
    print(f"output: {args.output}")
    print(f"input rows: {len(rows)}")
    print(f"summary rows: {len(summary)}")
    sys.exit(0)


if __name__ == "__main__":
    main()
