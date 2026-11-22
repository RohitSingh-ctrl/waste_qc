"""TrashNet helpers: folder scan (class-per-directory) and download."""
from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from src.utils import ensure_dir, get_logger

logger = get_logger(__name__)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def scan_trashnet(root: Path | str, class_names: Sequence[str]) -> Tuple[List[Path], List[int]]:
    """List ``(paths, labels)`` for a ``root/<class>/*.jpg`` folder dataset (labels follow ``class_names``).

    Handles the official zip layout ``dataset-resized/<class>/...`` automatically.
    """
    root = Path(root)
    if (root / "dataset-resized").is_dir():
        root = root / "dataset-resized"
    paths: List[Path] = []
    labels: List[int] = []
    for ci, name in enumerate(class_names):
        cdir = root / name
        if not cdir.is_dir():
            continue
        for p in sorted(cdir.iterdir()):
            if p.suffix.lower() in IMG_EXTS:
                paths.append(p)
                labels.append(ci)
    return paths, labels


def download_trashnet(dest: Path | str, url: str) -> Optional[Path]:
    """Download and unzip TrashNet; returns the dataset root or ``None`` on failure."""
    dest = ensure_dir(dest)
    zip_path = dest / "dataset-resized.zip"
    try:
        if not zip_path.exists():
            logger.info("Downloading TrashNet from %s", url)
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as r, open(zip_path, "wb") as fh:
                fh.write(r.read())
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest)
        return dest
    except Exception as exc:
        logger.warning("TrashNet download failed: %s", exc)
        return None
