"""Shared utilities: config loading, deterministic seeding, logging, figure saving.

Every module in the project saves plots through :func:`save_figure`, so all charts
land in ``outputs/figures/`` for the technical report and presentation.
"""
from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional, Union

import matplotlib

matplotlib.use("Agg")  # headless-safe backend (CI, servers)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "configs" / "config.yaml"

PathLike = Union[str, Path]


def load_config(path: Optional[PathLike] = None) -> Dict[str, Any]:
    """Load the YAML configuration file.

    Args:
        path: Optional path to a YAML file. Defaults to ``configs/config.yaml``.

    Returns:
        The configuration as a nested dictionary.
    """
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_path(p: PathLike) -> Path:
    """Resolve a config-relative path against the project root."""
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def ensure_dir(p: PathLike) -> Path:
    """Create a directory (and parents) if missing and return it."""
    path = resolve_path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int = 42, deterministic_torch: bool = True) -> None:
    """Seed Python, NumPy and (if installed) PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:  # torch is optional for the numpy-only modules
        pass


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a console logger with a compact format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


def figures_dir(cfg: Optional[Dict[str, Any]] = None) -> Path:
    """Directory where every chart is saved (``outputs/figures`` by default)."""
    rel = (cfg or {}).get("paths", {}).get("figures", "outputs/figures") if cfg else "outputs/figures"
    return ensure_dir(rel)


def save_figure(fig: plt.Figure, name: str, subdir: str = "", dpi: int = 150,
                cfg: Optional[Dict[str, Any]] = None) -> Path:
    """Save a matplotlib figure into ``outputs/figures`` and close it.

    Args:
        fig: Figure to save.
        name: File name (``.png`` appended if no extension is given).
        subdir: Optional sub-folder below ``outputs/figures``.
        dpi: Resolution.
        cfg: Optional config (to honour a custom figures path).

    Returns:
        Absolute path of the saved file.
    """
    out_dir = figures_dir(cfg)
    if subdir:
        out_dir = ensure_dir(out_dir / subdir)
    fname = name if Path(name).suffix else f"{name}.png"
    out_path = out_dir / fname
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_json(obj: Any, name: str, subdir: str = "", cfg: Optional[Dict[str, Any]] = None) -> Path:
    """Persist a JSON-serialisable object into ``outputs/logs``."""
    rel = (cfg or {}).get("paths", {}).get("logs", "outputs/logs") if cfg else "outputs/logs"
    out_dir = ensure_dir(rel)
    if subdir:
        out_dir = ensure_dir(out_dir / subdir)
    out_path = out_dir / (name if name.endswith(".json") else f"{name}.json")

    def _default(o: Any) -> Any:
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_default)
    return out_path
