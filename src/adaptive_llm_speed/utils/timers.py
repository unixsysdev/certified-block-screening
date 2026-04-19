"""CUDA / HIP event timers for wall-clock measurement.

Usage:
    t = CudaTimer()
    for _ in range(warmup):
        fn()
    with t:
        fn()
    elapsed_ms = t.elapsed_ms

    # or: many shots
    stats = time_many(fn, warmup=3, iters=10)
"""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

import torch


@dataclass
class TimingStats:
    n: int
    mean_ms: float
    median_ms: float
    std_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class CudaTimer:
    def __init__(self) -> None:
        self._use_cuda = torch.cuda.is_available()
        self._start = None
        self._end = None
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> "CudaTimer":
        if self._use_cuda:
            torch.cuda.synchronize()
            self._start = torch.cuda.Event(enable_timing=True)
            self._end = torch.cuda.Event(enable_timing=True)
            self._start.record()
        else:
            self._start = time.perf_counter_ns()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._use_cuda:
            self._end.record()
            torch.cuda.synchronize()
            self.elapsed_ms = self._start.elapsed_time(self._end)
        else:
            self.elapsed_ms = (time.perf_counter_ns() - self._start) / 1e6


def time_many(fn, *, warmup: int = 3, iters: int = 10) -> TimingStats:
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(iters):
        t = CudaTimer()
        with t:
            fn()
        samples.append(t.elapsed_ms)
    samples.sort()
    n = len(samples)
    p95_idx = max(0, int(round(0.95 * (n - 1))))
    return TimingStats(
        n=n,
        mean_ms=statistics.fmean(samples),
        median_ms=statistics.median(samples),
        std_ms=statistics.pstdev(samples) if n > 1 else 0.0,
        p95_ms=samples[p95_idx],
        min_ms=samples[0],
        max_ms=samples[-1],
    )
