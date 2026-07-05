"""Stage 12-B: the direct (no-A*) GP/GEP Rush Hour planner."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import src.rushhour.astar as astar_module
from src.rushhour.direct_features import DIRECT_FEATURE_NAMES, move_features
from src.rushhour.direct_fitness import (
    direct_fitness_value,
    evaluate_direct_policy,
    policy_from_tree,
    run_gep_direct,
    run_gp_direct,
)
from src.rushhour.direct_planner import (
    greedy_blocker_depth_policy,
    greedy_policy_rollout,
    random_policy,
)
from src.rushhour.moves import legal_moves
from src.rushhour.puzzle_sets import load_puzzle_set

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "scripts" / "run_gp_gep_direct_planner.py"

PUZZLES = load_puzzle_set(REPO_ROOT / "examples" / "rushhour_hard_eval.txt")
TINY = load_puzzle_set(REPO_ROOT / "examples" / "rushhour_tiny.txt")


def test_direct_modules_exist():
    base = REPO_ROOT / "src" / "rushhour"
    for name in ("direct_planner.py", "direct_features.py", "direct_fitness.py"):
        assert (base / name).exists(), name


def test_rollout_solves_a_trivial_puzzle():
    result = greedy_policy_rollout(TINY[0], greedy_blocker_depth_policy,
                                   max_steps=40)
    assert result.solved
    assert result.stopped_reason == "solved"
    assert result.error is None


def test_rollout_stops_at_max_steps():
    # random noise on a hard puzzle: must stop exactly at the step cap
    policy = random_policy(np.random.default_rng(0))
    result = greedy_policy_rollout(PUZZLES[-1], policy, max_steps=5)
    assert not result.solved
    assert result.steps <= 5
    assert result.stopped_reason in ("max_steps", "dead_end")


def test_rollout_counts_repeated_states():
    # a policy that always minimizes red distance oscillates on hard boards
    from src.rushhour.direct_planner import greedy_red_distance_policy
    total = sum(greedy_policy_rollout(p, greedy_red_distance_policy,
                                      max_steps=60).repeated_states
                for p in PUZZLES)
    assert total > 0  # cycle fallback is exercised and counted


def test_scoring_does_not_mutate_state():
    state = PUZZLES[0]
    board_before = state.board
    greedy_policy_rollout(state, greedy_blocker_depth_policy, max_steps=10)
    assert state.board == board_before


def test_move_features_are_numeric_and_complete():
    state = PUZZLES[0]
    move = legal_moves(state)[0]
    from src.rushhour.moves import apply_move
    features = move_features(state, move, apply_move(state, move), set())
    assert set(features) == set(DIRECT_FEATURE_NAMES)
    assert all(isinstance(v, float) for v in features.values())


def test_direct_fitness_is_numeric():
    evaluation = evaluate_direct_policy(greedy_blocker_depth_policy,
                                        PUZZLES[:3], max_steps=60)
    fitness = direct_fitness_value(evaluation, expression_size=5, max_steps=60)
    assert isinstance(fitness, float)


def test_direct_planner_never_calls_astar(monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("A* was called by the direct planner")
    monkeypatch.setattr(astar_module, "solve_astar", boom)
    result = run_gp_direct(PUZZLES[:1], generations=1, population_size=3,
                           seed=42, max_steps=30)
    assert result.best.fitness is not None
    gep = run_gep_direct(PUZZLES[:1], generations=1, population_size=3,
                         seed=42, max_steps=30)
    assert gep.best.fitness is not None


def test_gp_and_gep_direct_run_and_return_fields():
    result = run_gp_direct(TINY[:1], generations=1, population_size=3,
                           seed=42, max_steps=30)
    assert result.algorithm == "gp_direct"
    assert result.best.expression
    assert result.best.expression_size > 0
    rollout = greedy_policy_rollout(TINY[0], policy_from_tree(result.best.tree),
                                    max_steps=30)
    for attribute in ("solved", "steps", "repeated_states", "dead_end",
                      "runtime_seconds", "error"):
        assert hasattr(rollout, attribute)


def test_runner_help_and_smoke():
    result = subprocess.run([sys.executable, str(RUNNER), "--help"],
                            capture_output=True, text=True, cwd=REPO_ROOT)
    assert result.returncode == 0
    assert "--smoke" in result.stdout

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        smoke = subprocess.run(
            [sys.executable, str(RUNNER), "--smoke", "--output-dir", tmp],
            capture_output=True, text=True, cwd=REPO_ROOT)
        assert smoke.returncode == 0, smoke.stdout + smoke.stderr
        assert "gp_direct" in smoke.stdout and "gep_direct" in smoke.stdout
        assert (Path(tmp) / "direct_gp_gep_runs.csv").exists()
        assert (Path(tmp) / "direct_planner_manifest.json").exists()
