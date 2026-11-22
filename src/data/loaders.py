"""Loading real image arrays for the from-scratch modules (math sandbox, baseline MLP).

Real TrashNet is required by default. Synthetic data is only used when ``allow_synthetic=True``
(unit tests / CI) and is never substituted silently.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image

from src.data.synthetic import make_classification_arrays
from src.data.trashnet import scan_trashnet
from src.utils import load_config, resolve_path

INSTRUCTIONS = (
    "Real dataset not found. Download it first:\n"
    "  python -m src.data.pipeline --source trashnet --download\n"
    "or place the class folders (cardboard/glass/metal/paper/plastic/trash) in data/raw/trashnet/.\n"
    "(For TACO: --source taco --download, or put annotations.json + images in data/raw/taco/.)"
)


class DataNotFoundError(FileNotFoundError):
    """Raised when a required real dataset is missing."""

    def __init__(self, msg: str = "") -> None:
        super().__init__(f"{msg}\n{INSTRUCTIONS}".strip())


def _read(paths, size: int) -> np.ndarray:
    return np.stack([np.asarray(Image.open(p).convert("RGB").resize((size, size), Image.BILINEAR)) for p in paths])


def load_trashnet_arrays(cfg: Optional[Dict[str, Any]] = None, size: int = 32, split: Optional[str] = None,
                         per_class: Optional[int] = None, allow_synthetic: bool = False, seed: int = 42
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(images uint8 [N,size,size,3], labels)`` from real TrashNet.

    Uses the leakage-safe processed split (``data/processed/trashnet/classification/<split>``) when the
    pipeline has been run and ``split`` is given; otherwise reads the raw folder.

    Raises:
        DataNotFoundError: if no real images exist and ``allow_synthetic`` is False.
    """
    cfg = cfg or load_config()
    names = cfg["classes"]["names"]
    processed = resolve_path(cfg["paths"]["processed_root"]) / "trashnet" / "classification"
    root = processed / split if (split and (processed / split).is_dir()) else resolve_path(cfg["paths"]["raw_trashnet"])
    paths, labels = scan_trashnet(root, names)
    if not paths:
        if allow_synthetic:
            return make_classification_arrays(per_class or 60, size, names, seed)
        raise DataNotFoundError(f"No TrashNet images in {root}")
    paths, labels = np.asarray(paths, dtype=object), np.asarray(labels)
    rng = np.random.default_rng(seed)
    if per_class:
        keep = np.concatenate([rng.permutation(np.where(labels == c)[0])[:per_class] for c in np.unique(labels)])
        paths, labels = paths[keep], labels[keep]
    order = rng.permutation(len(paths))
    return _read(paths[order], size), labels[order].astype(np.int64)
