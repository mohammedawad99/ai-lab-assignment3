"""Check whether the official CVRP benchmark data is in place.

Usage:
    python scripts/check_official_cvrp_data.py [--data-dir data/official_cvrp]
        [--bks data/cvrp_bks.csv] [--strict] [--csv-output results/official_cvrp_data_check.csv]
"""

import argparse
import csv
import sys
from pathlib import Path

# make "src" importable when the script is run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments.official_benchmarks import (
    DEFAULT_BKS_PATH,
    DEFAULT_OFFICIAL_CVRP_DIR,
    check_official_data,
    expected_instance_path,
)


def main():
    parser = argparse.ArgumentParser(description="Check official CVRP data readiness.")
    parser.add_argument("--data-dir", default=DEFAULT_OFFICIAL_CVRP_DIR)
    parser.add_argument("--bks", default=DEFAULT_BKS_PATH)
    parser.add_argument("--strict", action="store_true",
                        help="all 6 official .vrp files must be present")
    parser.add_argument("--csv-output", help="optional CSV with one row per instance")
    args = parser.parse_args()

    report = check_official_data(data_dir=args.data_dir, bks_path=args.bks,
                                 strict=args.strict)

    print(f"data_dir: {report['data_dir']}")
    print(f"bks_path: {report['bks_path']}")
    print(f"ok: {'yes' if report['ok'] else 'no'}")
    print(f"expected instances: {', '.join(report['expected_instances'])}")
    print(f"found instances: {', '.join(report['found_instances']) or '(none)'}")
    print(f"missing instances: {', '.join(report['missing_instances']) or '(none)'}")
    print(f"unexpected files: {', '.join(report['unexpected_files']) or '(none)'}")
    print("bks values:")
    for name, cost in report["bks"].items():
        print(f"  {name}: {cost:g}")
    for error in report["errors"]:
        print(f"  error: {error}")

    if args.csv_output:
        csv_path = Path(args.csv_output)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["instance", "expected_path", "exists", "bks_cost"])
            for name in report["expected_instances"]:
                path = expected_instance_path(name, args.data_dir)
                writer.writerow([name, str(path), path.exists(),
                                 report["bks"].get(name, "")])
        print(f"csv written to: {csv_path}")

    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
