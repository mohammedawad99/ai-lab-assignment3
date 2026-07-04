"""Generate report tables and plots from existing experiment CSV files.

Usage:
    python scripts/generate_report_assets.py \
        --cvrp-raw results/smoke_suite/raw/cvrp_results.csv \
        --cvrp-summary results/smoke_suite/summary/cvrp_summary.csv \
        --ackley-raw results/smoke_suite/raw/ackley_results.csv \
        --ackley-summary results/smoke_suite/summary/ackley_summary.csv \
        --output-dir results/report_assets
"""

import argparse
import sys
from pathlib import Path

# make "src" importable when the script is run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments.report_assets import generate_report_assets


def main():
    parser = argparse.ArgumentParser(
        description="Generate report tables/plots from experiment CSVs."
    )
    parser.add_argument("--cvrp-raw")
    parser.add_argument("--cvrp-summary")
    parser.add_argument("--ackley-raw")
    parser.add_argument("--ackley-summary")
    parser.add_argument("--gp-gep-csv")
    parser.add_argument("--gp-gep-summary")
    parser.add_argument("--output-dir", default="results/report_assets")
    args = parser.parse_args()

    created = generate_report_assets(
        cvrp_raw=args.cvrp_raw,
        cvrp_summary=args.cvrp_summary,
        ackley_raw=args.ackley_raw,
        ackley_summary=args.ackley_summary,
        gp_gep_csv=args.gp_gep_csv,
        gp_gep_summary=args.gp_gep_summary,
        output_dir=args.output_dir,
    )

    print(f"output_dir: {args.output_dir}")
    print(f"assets created: {len(created)}")
    for path in created:
        print(f"  {path}")
    sys.exit(0)


if __name__ == "__main__":
    main()
