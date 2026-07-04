"""Helper to measure elapsed (wall clock) and CPU time of a function call."""

import time
from dataclasses import dataclass


@dataclass
class TimerResult:
    elapsed_time: float
    cpu_time: float


def measure_time(func, *args, **kwargs):
    """Run func(*args, **kwargs) and return (result, TimerResult)."""
    start_elapsed = time.perf_counter()
    start_cpu = time.process_time()
    result = func(*args, **kwargs)
    timer = TimerResult(
        elapsed_time=time.perf_counter() - start_elapsed,
        cpu_time=time.process_time() - start_cpu,
    )
    return result, timer
