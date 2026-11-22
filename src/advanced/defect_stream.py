"""Defect-rate time series for the LSTM monitor.

TrashNet/TACO are static image sets, so they contain no conveyor-belt history. Two sources are supported:

* **Real log (preferred):** a CSV with a ``defect_rate`` column, or ``defects`` + ``items`` columns.
* **Simulated stream (opt-in default for demos):** per-batch contamination rate with a shift cycle, slow drift and
  contamination *events* (ramp -> plateau -> decay). It is clearly simulated and never presented as measured data.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Union

import numpy as np
import pandas as pd


def simulate_defect_series(n_batches: int = 1500, items_per_batch: int = 50, base_rate: float = 0.12,
                           seed: int = 42) -> Dict[str, np.ndarray]:
    """Simulate the per-batch defect (contamination) rate of a sorting line.

    Returns:
        ``{"rate": observed rate per batch, "true_p": latent probability, "event": 1 inside a contamination event}``.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_batches)
    p = base_rate + 0.025 * np.sin(2 * np.pi * t / 96) + 0.02 * np.sin(2 * np.pi * t / 700 + rng.uniform(0, 6.28))
    event = np.zeros(n_batches, dtype=np.int8)
    pos = int(rng.integers(60, 100))
    while pos < n_batches - 40:
        ramp, plateau, decay = int(rng.integers(4, 9)), int(rng.integers(4, 10)), int(rng.integers(3, 7))
        amp = float(rng.uniform(0.25, 0.38))
        shape = np.concatenate([np.linspace(0, 1, ramp, endpoint=False), np.ones(plateau), np.linspace(1, 0, decay)])
        end = min(pos + len(shape), n_batches)
        p[pos:end] += amp * shape[: end - pos]
        event[pos:end] = 1
        pos += len(shape) + int(rng.integers(70, 140))
    p = np.clip(p, 0.01, 0.9)
    rate = rng.binomial(items_per_batch, p) / items_per_batch
    return {"rate": rate.astype(np.float32), "true_p": p.astype(np.float32), "event": event}


def load_defect_csv(path: Union[str, Path]) -> np.ndarray:
    """Read a real defect log. Needs ``defect_rate`` or both ``defects`` and ``items`` columns."""
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}
    if "defect_rate" in cols:
        rate = df[cols["defect_rate"]].astype(float).to_numpy()
    elif "defects" in cols and "items" in cols:
        rate = (df[cols["defects"]].astype(float) / df[cols["items"]].astype(float).clip(lower=1)).to_numpy()
    else:
        raise ValueError("CSV needs a 'defect_rate' column, or 'defects' and 'items' columns")
    rate = rate[np.isfinite(rate)]
    if len(rate) == 0:
        raise ValueError("CSV has no valid rows")
    return np.clip(rate, 0.0, 1.0).astype(np.float32)
