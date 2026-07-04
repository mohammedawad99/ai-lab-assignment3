"""Safe evaluation of Rush Hour heuristics.

This is the safety layer for the later GP/GEP stages: every A* call has a
node cap and a time cap, the whole evaluation has a total time budget, and a
heuristic that raises an exception cannot crash or hang the evaluation.
"""

import csv
import time
from dataclasses import dataclass
from pathlib import Path

from src.rushhour.astar import solve_astar
from src.rushhour.model import RushHourState


@dataclass
class PuzzleEvaluation:
    puzzle_index: int
    solved: bool
    cost: int
    expanded_nodes: int
    generated_nodes: int
    elapsed_time: float
    cpu_time: float
    stopped_reason: str | None


@dataclass
class HeuristicEvaluation:
    heuristic_name: str
    puzzle_count: int
    solved_count: int
    total_cost: int
    total_expanded_nodes: int
    total_generated_nodes: int
    total_elapsed_time: float
    total_cpu_time: float
    timeout_count: int
    max_nodes_count: int
    no_solution_count: int
    puzzle_results: list[PuzzleEvaluation]

    @property
    def solved_rate(self) -> float:
        if self.puzzle_count == 0:
            return 0.0
        return self.solved_count / self.puzzle_count


def evaluate_heuristic(heuristic_name, heuristic_func, puzzles: list[RushHourState],
                       max_nodes_per_puzzle: int = 10000,
                       max_time_per_puzzle_sec: float = 1.0,
                       max_total_time_sec: float = 10.0) -> HeuristicEvaluation:
    """Run A* with the heuristic on every puzzle, with node and time caps."""
    start_total = time.perf_counter()
    start_cpu = time.process_time()
    results: list[PuzzleEvaluation] = []

    for index, puzzle in enumerate(puzzles):
        remaining = max_total_time_sec - (time.perf_counter() - start_total)
        if remaining <= 0:
            # total budget exhausted: count the rest as timeouts without running A*
            results.append(PuzzleEvaluation(
                puzzle_index=index, solved=False, cost=0,
                expanded_nodes=0, generated_nodes=0,
                elapsed_time=0.0, cpu_time=0.0, stopped_reason="timeout",
            ))
            continue

        try:
            search = solve_astar(
                puzzle,
                heuristic=heuristic_func,
                max_nodes=max_nodes_per_puzzle,
                timeout_sec=min(max_time_per_puzzle_sec, remaining),
            )
            results.append(PuzzleEvaluation(
                puzzle_index=index,
                solved=search.solved,
                cost=search.cost,
                expanded_nodes=search.expanded_nodes,
                generated_nodes=search.generated_nodes,
                elapsed_time=search.elapsed_time,
                cpu_time=search.cpu_time,
                stopped_reason=search.stopped_reason,
            ))
        except Exception:
            # a broken heuristic must not stop the whole evaluation
            results.append(PuzzleEvaluation(
                puzzle_index=index, solved=False, cost=0,
                expanded_nodes=0, generated_nodes=0,
                elapsed_time=0.0, cpu_time=0.0, stopped_reason="heuristic_error",
            ))

    reasons = [r.stopped_reason for r in results]
    return HeuristicEvaluation(
        heuristic_name=heuristic_name,
        puzzle_count=len(puzzles),
        solved_count=sum(1 for r in results if r.solved),
        total_cost=sum(r.cost for r in results),
        total_expanded_nodes=sum(r.expanded_nodes for r in results),
        total_generated_nodes=sum(r.generated_nodes for r in results),
        total_elapsed_time=time.perf_counter() - start_total,
        total_cpu_time=time.process_time() - start_cpu,
        timeout_count=reasons.count("timeout"),
        max_nodes_count=reasons.count("max_nodes"),
        no_solution_count=reasons.count("no_solution"),
        puzzle_results=results,
    )


def fitness_from_evaluation(evaluation: HeuristicEvaluation) -> float:
    """Simple default score for later GP/GEP fitness. Higher is better."""
    return float(
        evaluation.solved_count * 10000
        - evaluation.total_expanded_nodes
        - evaluation.total_cost * 10
        - evaluation.timeout_count * 2000
        - evaluation.max_nodes_count * 1000
        - evaluation.no_solution_count * 1000
    )


CSV_HEADER = [
    "heuristic_name", "puzzle_count", "solved_count", "solved_rate",
    "total_cost", "total_expanded_nodes", "total_generated_nodes",
    "total_elapsed_time", "total_cpu_time",
    "timeout_count", "max_nodes_count", "no_solution_count", "fitness",
]


def write_evaluation_csv(evaluations: list[HeuristicEvaluation], path) -> None:
    """One CSV row per heuristic evaluation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        for ev in evaluations:
            writer.writerow([
                ev.heuristic_name, ev.puzzle_count, ev.solved_count,
                f"{ev.solved_rate:.4f}", ev.total_cost,
                ev.total_expanded_nodes, ev.total_generated_nodes,
                f"{ev.total_elapsed_time:.6f}", f"{ev.total_cpu_time:.6f}",
                ev.timeout_count, ev.max_nodes_count, ev.no_solution_count,
                fitness_from_evaluation(ev),
            ])
