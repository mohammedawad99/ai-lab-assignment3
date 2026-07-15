"""Tests for the final experiment plan and the report skeleton."""

import subprocess
import sys
from pathlib import Path

from src.experiments.final_plan import (
    REQUIRED_SIX,
    build_ackley_commands,
    build_all_commands,
    build_cvrp_commands,
    build_rushhour_commands,
    load_final_plan,
    validate_final_plan,
    write_command_file,
)
from src.experiments.official_benchmarks import OFFICIAL_CVRP_INSTANCES

REPO_ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = REPO_ROOT / "configs" / "final_experiment_plan.json"
REPORT_PATH = REPO_ROOT / "report" / "assignment3_report.md"


def load_plan():
    return load_final_plan(PLAN_PATH)


# ---------- plan ----------

def test_plan_loads_and_validates():
    plan = load_plan()
    ok, errors = validate_final_plan(plan, require_official_data=False)
    assert ok, errors


def test_plan_instances_and_algorithms():
    plan = load_plan()
    assert plan["cvrp"]["instances"] == OFFICIAL_CVRP_INSTANCES
    assert plan["cvrp"]["required_algorithms"] == REQUIRED_SIX
    assert plan["ackley"]["required_algorithms"] == REQUIRED_SIX
    assert plan["ackley"]["dimension"] == 10


def test_command_builders():
    plan = load_plan()
    cvrp = build_cvrp_commands(plan)
    assert len(cvrp) == 6
    assert all("run_experiments.py --part cvrp" in c for c in cvrp)
    assert any("P-n16-k8" in c for c in cvrp)
    assert all("--bks data/cvrp_bks.csv" in c for c in cvrp)

    ackley = build_ackley_commands(plan)
    assert len(ackley) == 1
    assert "--dimension 10" in ackley[0]

    rushhour = build_rushhour_commands(plan)
    assert len(rushhour) == 1
    assert "compare_gp_gep_rushhour.py" in rushhour[0]


def test_build_all_commands_and_write(tmp_path):
    plan = load_plan()
    commands = build_all_commands(plan)
    assert set(commands) == {"cvrp", "ackley", "rushhour", "assets"}
    assert commands["assets"]  # summaries + report assets

    path = write_command_file(commands, tmp_path / "sub" / "commands.txt")
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "# ---- cvrp ----" in text
    assert "# ---- assets ----" in text


# ---------- CLI ----------

def run_cli(*extra_args):
    script = REPO_ROOT / "scripts" / "print_final_experiment_plan.py"
    return subprocess.run(
        [sys.executable, str(script), *extra_args],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )


def test_cli_prints_plan():
    proc = run_cli()
    assert proc.returncode == 0
    assert "validation: ok" in proc.stdout
    assert "cvrp commands: 6" in proc.stdout


def test_cli_writes_commands(tmp_path):
    path = tmp_path / "commands.txt"
    proc = run_cli("--write-commands", str(path))
    assert proc.returncode == 0
    assert path.exists()


def test_cli_require_official_data_when_present():
    data_dir = REPO_ROOT / "data" / "official_cvrp"
    all_present = all((data_dir / f"{name}.vrp").exists()
                      for name in OFFICIAL_CVRP_INSTANCES)
    proc = run_cli("--require-official-data")
    if all_present:
        assert proc.returncode == 0
    else:
        assert proc.returncode == 1  # missing data is a validation failure


# ---------- report skeleton ----------

def test_report_skeleton_exists_with_sections():
    text = REPORT_PATH.read_text(encoding="utf-8")
    for heading in [
        "# Assignment 3 Report",
        "## 1. Introduction",
        "## 2. Implementation Overview",
        "## 3. Part A — Ackley Function",
        "## 4. Part A — CVRP",
        "## 5. Multi-stage CVRP Heuristic",
        "## 6. CVRP Algorithms",
        "## 7. Ackley Adaptations",
        "## 8. Part B — Rush Hour with GP and GEP",
        "## 9. Results",
        "## 10. Analysis and Discussion",
        "## 11. Complexity and Practical Considerations",
        "## 12. Use of AI Tools",
        "## 13. Reproducibility",
        "## 14. Conclusion",
    ]:
        assert heading in text, heading


def test_report_skeleton_placeholders_and_facts():
    text = REPORT_PATH.read_text(encoding="utf-8")
    # since Stage 9-C the report is filled, so no placeholders remain
    assert "[fill after final run" not in text
    assert "d = 10" in text
    assert "[-32.768, 32.768]" in text
    for name in OFFICIAL_CVRP_INSTANCES:
        assert name in text
    assert "AI tools were used" in text


def test_report_skeleton_makes_no_final_claims():
    text = REPORT_PATH.read_text(encoding="utf-8").lower()
    for phrase in ["we prove", "best overall", "final benchmark shows"]:
        assert phrase not in text
