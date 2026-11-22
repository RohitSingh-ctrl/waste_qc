"""TACO (Trash Annotations in Context) helpers: category mapping, parsing, download."""
from __future__ import annotations

import json
import http.client
import os
import re
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image

from src.data.coco import CategoryMapper, DetectionSample, load_coco_samples
from src.utils import ensure_dir, get_logger

logger = get_logger(__name__)


def build_taco_mapper(class_names: Sequence[str], keyword_map: Dict[str, Sequence[str]],
                      default: str = "trash") -> CategoryMapper:
    """Create a ``(name, supercategory) -> unified idx`` mapper from ordered keyword rules.

    Exact unified-class names (as used by the synthetic data) are matched first; otherwise the
    ordered ``keyword_map`` is scanned on ``"<name> <supercategory>"`` (word-prefix regex).
    Unmatched categories fall back to ``default`` (TACO has many non-recyclable items).
    """
    idx = {n: i for i, n in enumerate(class_names)}
    rules: List[Tuple[int, "re.Pattern[str]"]] = []
    for cls, kws in keyword_map.items():
        for kw in kws:
            rules.append((idx[cls], re.compile(r"\b" + re.escape(kw.lower()))))

    def mapper(name: str, supercategory: str = "") -> Optional[int]:
        if name in idx:
            return idx[name]
        text = f"{name} {supercategory}".lower()
        for ci, pat in rules:
            if pat.search(text):
                return ci
        return idx[default]

    return mapper


def load_taco(annotations_json: Path | str, images_root: Path | str, class_names: Sequence[str],
              keyword_map: Dict[str, Sequence[str]], require_exists: bool = False
              ) -> Tuple[List[DetectionSample], Dict[str, Any]]:
    """Parse TACO ``annotations.json`` into unified :class:`DetectionSample` objects."""
    return load_coco_samples(annotations_json, images_root, build_taco_mapper(class_names, keyword_map),
                             require_exists=require_exists)


def _is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
    except (OSError, SyntaxError, ValueError):
        return False
    return True


def _fetch(url: str, dest: Path, timeout: int = 60, validate_image: bool = False) -> bool:
    temporary_path: Optional[Path] = None
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            with tempfile.NamedTemporaryFile(
                dir=dest.parent, prefix=f".{dest.name}.", suffix=".part", delete=False
            ) as fh:
                temporary_path = Path(fh.name)
                while chunk := response.read(1024 * 1024):
                    fh.write(chunk)
        if validate_image and not _is_valid_image(temporary_path):
            raise ValueError("downloaded response is not a valid image")
        os.replace(temporary_path, dest)
        temporary_path = None
        return True
    except (http.client.HTTPException, OSError, SyntaxError, ValueError) as exc:
        logger.warning("Download failed (%s): %s", url, exc)
        return False
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def download_taco(dest: Path | str, annotations_url: str, max_images: Optional[int] = None) -> Optional[Path]:
    """Download TACO annotations and (optionally the first ``max_images``) images from Flickr URLs.

    Returns the annotations path, or ``None`` if the download failed (caller should fall back
    to the synthetic dataset).
    """
    from tqdm import tqdm

    dest = ensure_dir(dest)
    ann_path = dest / "annotations.json"
    if not ann_path.exists() and not _fetch(annotations_url, ann_path):
        return None
    data = json.loads(ann_path.read_text(encoding="utf-8"))
    imgs = data["images"][:max_images] if max_images else data["images"]
    for im in tqdm(imgs, desc="TACO images"):
        out = dest / im["file_name"]
        if out.exists():
            if _is_valid_image(out):
                continue
            logger.warning("Existing image is incomplete or invalid; downloading again: %s", out)
            out.unlink()
        ensure_dir(out.parent)
        url = im.get("flickr_url") or im.get("flickr_640_url")
        if url:
            _fetch(url, out, validate_image=True)
    return ann_path
