"""Latency / FPS / model-size benchmarking shared by every model (feeds the deployment recommendation)."""
from __future__ import annotations

import time
from typing import Any, Callable, Dict

import numpy as np


def count_parameters(model: Any) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def model_size_mb(model: Any) -> float:
    return float(sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 ** 2)


def measure_latency(fn: Callable[[], Any], warmup: int = 3, runs: int = 20, sync_cuda: bool = True) -> Dict[str, float]:
    """Time ``fn()`` (one single-image inference). Returns mean/p95 latency in ms and FPS."""
    try:
        import torch
        cuda = sync_cuda and torch.cuda.is_available()
    except ImportError:  # pragma: no cover
        cuda = False
    sync = (lambda: torch.cuda.synchronize()) if cuda else (lambda: None)
    for _ in range(warmup):
        fn()
    sync()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        sync()
        times.append((time.perf_counter() - t0) * 1000.0)
    mean = float(np.mean(times))
    return {"latency_ms_mean": mean, "latency_ms_p95": float(np.percentile(times, 95)), "fps": 1000.0 / max(mean, 1e-9)}


def peak_gpu_memory_mb() -> float:
    """Peak CUDA memory since process start (0.0 on CPU-only machines)."""
    try:
        import torch
        return float(torch.cuda.max_memory_allocated() / 1024 ** 2) if torch.cuda.is_available() else 0.0
    except ImportError:  # pragma: no cover
        return 0.0
