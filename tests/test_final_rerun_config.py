"""Tests for the Stage 10-C tuned final rerun wiring."""

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def run_script(script, *extra):
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / script), *extra],
        capture_output=True, text=True, cwd=REPO_ROOT)


def test_tuned_cvrp_config():
    config = json.loads((REPO_ROOT / "configs" / "tuned_cvrp_settings.json").read_text())
    assert config["alns"]["enhanced_operators"] is True
    assert config["sa"]["iterations_factor"] == 50
    assert config["ga_island"]["population_size"] == 30
    # the ALNS selection rule must be declared in the config, not applied ad hoc
    policy = config["alns_policy"]
    assert "M-n200-k17" in policy["enhanced_excluded_instances"]
    assert "declared before" in policy["main_result_rule"]


def test_hard_benchmark_config():
    config = json.loads(
        (REPO_ROOT / "configs" / "rushhour_hard_benchmark.json").read_text())
    assert config["eval_puzzles"] == "examples/rushhour_hard_eval.txt"
    assert config["seeds"] == [42, 43, 44]


def test_final_runner_supports_tuned_and_hard_options():
    proc = run_script("run_final_experiments.py", "--help")
    assert proc.returncode == 0
    assert "--tuned-cvrp" in proc.stdout
    assert "--rushhour-hard" in proc.stdout


def test_extractor_supports_help():
    proc = run_script("extract_final_results_v2.py", "--help")
    assert proc.returncode == 0
    assert "--results-dir" in proc.stdout


def test_alns_enhanced_variant_dispatch():
    # the experiment layer must run the enhanced variant under its own name
    from src.cvrp.io_cvrplib import parse_cvrplib
    from src.experiments.cvrp_experiments import run_one_cvrp_algorithm
    tiny = parse_cvrplib(REPO_ROOT / "examples" / "tiny_cvrp.vrp")
    tuned = json.loads((REPO_ROOT / "configs" / "tuned_cvrp_settings.json").read_text())
    row = run_one_cvrp_algorithm(tiny, "examples/tiny_cvrp.vrp", "alns_enhanced",
                                 42, 10, 5.0, tuned=tuned)
    assert row["algorithm"] == "cvrp_alns_enhanced"
    assert row["feasible"] is True


def test_no_generated_or_official_files_tracked():
    tracked = subprocess.run(["git", "ls-files"], capture_output=True,
                             text=True, cwd=REPO_ROOT).stdout.splitlines()
    assert not any(p.startswith("results/") and p != "results/.gitkeep"
                   for p in tracked)
    assert not any(p.endswith(".vrp") and "official_cvrp" in p for p in tracked)


def test_bnb_small_instance_mode():
    # the tuned config gives small instances a deeper LDS budget; it must
    # stay feasible, respect the timeout, and lose no customers
    import time
    from src.cvrp.io_cvrplib import parse_cvrplib
    from src.cvrp.validate import validate_solution
    from src.experiments.cvrp_experiments import run_one_cvrp_algorithm

    tiny = parse_cvrplib(REPO_ROOT / "examples" / "tiny_cvrp.vrp")
    tuned = json.loads((REPO_ROOT / "configs" / "tuned_cvrp_settings.json").read_text())
    assert tuned["bnb_lds"]["small_instance_max_customers"] >= 4
    start = time.perf_counter()
    row = run_one_cvrp_algorithm(tiny, "examples/tiny_cvrp.vrp", "bnb_lds",
                                 42, 10, 5.0, tuned=tuned)
    assert time.perf_counter() - start < 6.0  # timeout respected with margin
    assert row["feasible"] is True
    assert float(row["best_cost"]) < 81.0


def test_bnb_deep_mode_respects_timeout_directly():
    from src.cvrp.io_cvrplib import parse_cvrplib
    from src.cvrp.solvers.bnb_lds import run_cvrp_bnb_lds
    from src.cvrp.validate import validate_solution
    import time
    inst = parse_cvrplib(REPO_ROOT / "data" / "official_cvrp" / "E-n22-k4.vrp") \
        if (REPO_ROOT / "data" / "official_cvrp" / "E-n22-k4.vrp").exists() else None
    if inst is None:
        import pytest
        pytest.skip("official E-n22-k4 not placed")
    start = time.perf_counter()
    result = run_cvrp_bnb_lds(inst, max_discrepancy=15, max_nodes=2000000,
                              seed=42, timeout_sec=3.0)
    assert time.perf_counter() - start < 10.0
    assert result.feasible
    check = validate_solution(inst, result.best_solution)
    assert check.feasible, check.errors
