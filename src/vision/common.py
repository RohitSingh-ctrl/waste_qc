"""Shared PyTorch helpers for the vision models."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from src.data.loaders import DataNotFoundError
from src.utils import load_config, resolve_path

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def processed_dir(cfg: Dict[str, Any], tag: str, sub: str = "") -> Path:
    """``data/processed/<tag>/<sub>`` resolved against the project root (or the Drive path set in config)."""
    p = resolve_path(cfg["paths"]["processed_root"]) / tag / sub
    if not p.is_dir():
        raise DataNotFoundError(f"Processed data not found at {p}. Run the data pipeline first")
    return p


def classification_root(cfg: Dict[str, Any], synthetic: bool = False) -> Path:
    """TrashNet folders if available, else TACO crops (or the synthetic crops when ``synthetic``)."""
    if synthetic:
        return processed_dir(cfg, "synthetic_taco", "crops")
    root = resolve_path(cfg["paths"]["processed_root"])
    for tag, sub in (("trashnet", "classification"), ("taco", "crops")):
        if (root / tag / sub).is_dir():
            return root / tag / sub
    raise DataNotFoundError(f"No processed classification data under {root}")


def detection_tag(cfg: Dict[str, Any], synthetic: bool = False) -> str:
    if synthetic:
        return "synthetic_taco"
    if (resolve_path(cfg["paths"]["processed_root"]) / "taco").is_dir():
        return "taco"
    raise DataNotFoundError("TACO not processed. Run: python -m src.data.pipeline --source taco --download")


class Timer:
    def __enter__(self) -> "Timer":
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *_: Any) -> None:
        self.seconds = time.perf_counter() - self.t0


def checkpoint_path(cfg: Dict[str, Any], name: str) -> Path:
    d = resolve_path(cfg["paths"]["checkpoints"])
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{name}.pt"


def parse_common_args(desc: str):
    import argparse
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--synthetic", action="store_true", help="use synthetic data (smoke tests only)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--image-size", type=int, default=None)
    ap.add_argument("--config", default=None)
    return ap
