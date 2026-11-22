"""Minimal dependency-free COCO parser (no pycocotools) shared by TACO and synthetic data."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

CategoryMapper = Callable[[str, str], Optional[int]]  # (name, supercategory) -> unified class idx or None


@dataclass
class DetectionSample:
    """One image with unified-class boxes/polygons.

    Attributes:
        boxes: ``(K, 4)`` float32 absolute ``xyxy``.
        labels: ``(K,)`` int64 unified class indices.
        polygons: per-instance list of flat ``[x0,y0,x1,y1,...]`` polygons (a list of rings).
    """

    image_id: int
    image_path: Path
    width: int
    height: int
    boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), np.float32))
    labels: np.ndarray = field(default_factory=lambda: np.zeros((0,), np.int64))
    polygons: List[List[List[float]]] = field(default_factory=list)
    scene_id: Optional[int] = None


def polygons_to_mask(polygons: Sequence[Sequence[float]], width: int, height: int) -> np.ndarray:
    """Rasterise polygon rings (flat coordinate lists) into a boolean mask."""
    m = Image.new("L", (width, height), 0)
    d = ImageDraw.Draw(m)
    for ring in polygons:
        if len(ring) >= 6:
            d.polygon([(ring[i], ring[i + 1]) for i in range(0, len(ring) - 1, 2)], fill=1)
    return np.asarray(m, dtype=bool)


def load_coco_samples(json_path: Path | str, images_root: Path | str, mapper: CategoryMapper,
                      require_exists: bool = False) -> Tuple[List[DetectionSample], Dict[str, Any]]:
    """Parse a COCO json into :class:`DetectionSample` objects with *unified* labels.

    Args:
        json_path: Path to ``annotations.json``.
        images_root: Folder that ``file_name`` entries are relative to.
        mapper: Maps ``(category name, supercategory)`` to a unified class index (or ``None`` to drop).
        require_exists: Skip images whose file is missing on disk (useful for partial TACO downloads).

    Returns:
        ``(samples, stats)`` where stats reports dropped annotations / missing files.
    """
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    cats = {c["id"]: (c["name"], c.get("supercategory", "")) for c in data["categories"]}
    by_img: Dict[int, List[Dict[str, Any]]] = {}
    for a in data["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    root = Path(images_root)
    samples: List[DetectionSample] = []
    stats = {"images": 0, "missing_files": 0, "annotations": 0, "dropped_annotations": 0}
    for im in data["images"]:
        path = root / im["file_name"]
        if require_exists and not path.exists():
            stats["missing_files"] += 1
            continue
        boxes, labels, polys = [], [], []
        for a in by_img.get(im["id"], []):
            name, sup = cats[a["category_id"]]
            u = mapper(name, sup)
            if u is None or a.get("iscrowd", 0):
                stats["dropped_annotations"] += 1
                continue
            x, y, w, h = a["bbox"]
            if w < 2 or h < 2:
                stats["dropped_annotations"] += 1
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(u)
            seg = a.get("segmentation", [])
            polys.append([list(map(float, r)) for r in seg] if isinstance(seg, list) else [])  # RLE unsupported -> empty
            stats["annotations"] += 1
        samples.append(DetectionSample(
            image_id=im["id"], image_path=path, width=im["width"], height=im["height"],
            boxes=np.asarray(boxes, np.float32).reshape(-1, 4), labels=np.asarray(labels, np.int64),
            polygons=polys, scene_id=im.get("scene_id")))
        stats["images"] += 1
    return samples, stats
