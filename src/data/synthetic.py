"""Lightweight synthetic waste dataset generator for CI, unit tests and demos.

Produces (a) conveyor-belt *scenes* with COCO boxes + polygon masks (TACO-like) and
(b) centred single-item images in ``class/`` folders (TrashNet-like). Each class has a
distinct colour/shape signature so even tiny models can learn it in seconds.
Fully deterministic given a seed.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from src.utils import ensure_dir, get_logger, set_seed

logger = get_logger(__name__)

DEFAULT_CLASSES: List[str] = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# base RGB colour and polygon family per class
CLASS_STYLE: Dict[str, Dict[str, Any]] = {
    "cardboard": {"color": (160, 115, 70), "shape": "rect"},
    "glass":     {"color": (90, 190, 190), "shape": "ellipse"},
    "metal":     {"color": (150, 150, 160), "shape": "circle"},
    "paper":     {"color": (235, 235, 225), "shape": "pentagon"},
    "plastic":   {"color": (60, 110, 210), "shape": "blob_round"},
    "trash":     {"color": (50, 70, 40), "shape": "blob_rough"},
}

Polygon = List[Tuple[float, float]]


# --------------------------------------------------------------------------- geometry
def _rotate(points: Sequence[Tuple[float, float]], cx: float, cy: float, ang: float) -> Polygon:
    c, s = math.cos(ang), math.sin(ang)
    return [(cx + (x - cx) * c - (y - cy) * s, cy + (x - cx) * s + (y - cy) * c) for x, y in points]


def make_polygon(shape: str, cx: float, cy: float, r: float, rng: np.random.Generator) -> Polygon:
    """Return a polygon (list of (x, y)) of a given family centred at ``(cx, cy)``."""
    ang = float(rng.uniform(0, math.pi))
    if shape == "rect":
        w, h = r * rng.uniform(1.0, 1.4), r * rng.uniform(0.6, 0.9)
        pts = [(cx - w, cy - h), (cx + w, cy - h), (cx + w, cy + h), (cx - w, cy + h)]
        return _rotate(pts, cx, cy, ang)
    if shape == "ellipse":
        a, b = r * 0.55, r * 1.1
        pts = [(cx + a * math.cos(t), cy + b * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 28, endpoint=False)]
        return _rotate(pts, cx, cy, ang)
    if shape == "circle":
        return [(cx + r * math.cos(t), cy + r * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 32, endpoint=False)]
    if shape == "pentagon":
        pts = [(cx + r * math.cos(t), cy + r * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 5, endpoint=False)]
        return _rotate(pts, cx, cy, ang)
    n = 24
    jitter = 0.12 if shape == "blob_round" else 0.45
    radii = r * (1.0 + rng.uniform(-jitter, jitter, size=n))
    return [(cx + rr * math.cos(t), cy + rr * math.sin(t)) for rr, t in zip(radii, np.linspace(0, 2 * math.pi, n, endpoint=False))]


def polygon_bbox(poly: Polygon, size: int) -> Tuple[float, float, float, float]:
    """Axis-aligned bbox ``(x, y, w, h)`` clipped to the image."""
    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
    x0, y0 = max(0.0, min(xs)), max(0.0, min(ys))
    x1, y1 = min(float(size), max(xs)), min(float(size), max(ys))
    return x0, y0, x1 - x0, y1 - y0


# --------------------------------------------------------------------------- rendering
def _background(size: int, rng: np.random.Generator) -> np.ndarray:
    """Dark-grey conveyor belt with faint horizontal ribs and sensor noise."""
    base = np.full((size, size, 3), 70, dtype=np.float32)
    ribs = (np.sin(np.arange(size) / 3.0) * 6.0)[:, None, None]
    return np.clip(base + ribs + rng.normal(0, 4, (size, size, 3)), 0, 255)


def _jitter_color(color: Tuple[int, int, int], rng: np.random.Generator, amount: int = 22) -> Tuple[int, int, int]:
    return tuple(int(np.clip(c + rng.integers(-amount, amount + 1), 0, 255)) for c in color)  # type: ignore[return-value]


def _draw_item(draw: ImageDraw.ImageDraw, label: str, poly: Polygon, rng: np.random.Generator) -> None:
    color = _jitter_color(CLASS_STYLE[label]["color"], rng)
    draw.polygon(poly, fill=color)
    if label in ("metal", "glass"):  # specular highlight
        xs, ys = [p[0] for p in poly], [p[1] for p in poly]
        cx, cy = float(np.mean(xs)), float(np.mean(ys))
        rr = 0.25 * (max(xs) - min(xs))
        hl = tuple(min(255, c + 70) for c in color)
        draw.ellipse([cx - rr * 1.2, cy - rr * 1.4, cx - rr * 0.2, cy - rr * 0.4], fill=hl)


def rasterize_polygon(poly: Polygon, size: int) -> np.ndarray:
    """Rasterise a polygon to a boolean ``(size, size)`` mask."""
    m = Image.new("L", (size, size), 0)
    ImageDraw.Draw(m).polygon(poly, fill=1)
    return np.asarray(m, dtype=bool)


def generate_scene(size: int, rng: np.random.Generator, classes: Sequence[str] = DEFAULT_CLASSES,
                   class_probs: Optional[Sequence[float]] = None, max_items: int = 4,
                   noise_std: float = 6.0) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Render a conveyor scene with 1..``max_items`` items.

    Returns:
        ``(image uint8 HxWx3, items)`` where each item has ``label``, ``polygon`` and ``bbox`` (x, y, w, h).
    """
    p = None if class_probs is None else np.asarray(class_probs, dtype=float) / np.sum(class_probs)
    bg = _background(size, rng)
    img = Image.fromarray(bg.astype(np.uint8))
    draw = ImageDraw.Draw(img)
    items: List[Dict[str, Any]] = []
    centers: List[Tuple[float, float, float]] = []
    for _ in range(int(rng.integers(1, max_items + 1))):
        label = str(rng.choice(list(classes), p=p))
        r = size * rng.uniform(0.09, 0.16)
        for _try in range(25):  # rejection sampling to limit overlaps
            cx, cy = rng.uniform(r * 1.5, size - r * 1.5, size=2)
            if all(math.hypot(cx - ox, cy - oy) > (r + orr) * 1.1 for ox, oy, orr in centers):
                break
        else:
            continue
        poly = make_polygon(CLASS_STYLE[label]["shape"], float(cx), float(cy), float(r), rng)
        _draw_item(draw, label, poly, rng)
        centers.append((float(cx), float(cy), float(r)))
        items.append({"label": label, "polygon": poly, "bbox": polygon_bbox(poly, size)})
    arr = np.asarray(img, dtype=np.float32) + rng.normal(0, noise_std, (size, size, 3))
    return np.clip(arr, 0, 255).astype(np.uint8), items


def render_single_item(label: str, size: int, rng: np.random.Generator, noise_std: float = 6.0) -> np.ndarray:
    """Centred, large single item on the belt (TrashNet-style classification sample)."""
    img = Image.fromarray(_background(size, rng).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    r = size * rng.uniform(0.16, 0.36)
    cx, cy = size / 2 + rng.uniform(-size * 0.20, size * 0.20, 2)
    poly = make_polygon(CLASS_STYLE[label]["shape"], float(cx), float(cy), float(r), rng)
    _draw_item(draw, label, poly, rng)
    arr = np.asarray(img, dtype=np.float32) + rng.normal(0, noise_std, (size, size, 3))
    return np.clip(arr, 0, 255).astype(np.uint8)


def _near_duplicate(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Slightly perturbed copy (brightness + tiny noise) -> perceptual-hash near-duplicate."""
    out = img.astype(np.float32) * rng.uniform(0.97, 1.03) + rng.normal(0, 1.5, img.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- in-memory arrays
def make_classification_arrays(n_per_class: int, size: int = 64, classes: Sequence[str] = DEFAULT_CLASSES,
                               seed: int = 42, noise_std: float = 6.0) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(images uint8 [N,H,W,3], labels int64 [N])`` shuffled deterministically."""
    rng = np.random.default_rng(seed)
    imgs, labels = [], []
    for ci, c in enumerate(classes):
        for _ in range(n_per_class):
            imgs.append(render_single_item(c, size, rng, noise_std))
            labels.append(ci)
    order = rng.permutation(len(labels))
    return np.stack(imgs)[order], np.asarray(labels, dtype=np.int64)[order]


# --------------------------------------------------------------------------- on-disk datasets
def write_synthetic_coco(root: Path | str, n_scenes: int = 120, size: int = 128, seed: int = 42,
                         classes: Sequence[str] = DEFAULT_CLASSES, max_items: int = 4,
                         duplicate_fraction: float = 0.1, noise_std: float = 6.0,
                         rare_classes: Sequence[str] = ("trash",), rare_weight: float = 0.5) -> Path:
    """Write a TACO-like COCO dataset (``images/`` + ``annotations.json``) and return the JSON path.

    Near-duplicate scenes (for dedup testing) are tagged with the same ``scene_id`` as their source.
    """
    set_seed(seed)
    rng = np.random.default_rng(seed)
    root = ensure_dir(root)
    img_dir = ensure_dir(root / "images")
    probs = [rare_weight if c in rare_classes else 1.0 for c in classes]
    cats = [{"id": i + 1, "name": c, "supercategory": c} for i, c in enumerate(classes)]
    images: List[Dict[str, Any]] = []
    anns: List[Dict[str, Any]] = []
    ann_id = 1

    def _add(arr: np.ndarray, items: List[Dict[str, Any]], scene_id: int) -> None:
        nonlocal ann_id
        img_id = len(images) + 1
        fname = f"scene_{img_id:05d}.png"
        Image.fromarray(arr).save(img_dir / fname)
        images.append({"id": img_id, "file_name": fname, "width": size, "height": size, "scene_id": scene_id})
        for it in items:
            x, y, w, h = it["bbox"]
            flat = [float(v) for pt in it["polygon"] for v in pt]
            anns.append({"id": ann_id, "image_id": img_id, "category_id": classes.index(it["label"]) + 1,
                         "bbox": [x, y, w, h], "area": float(w * h), "segmentation": [flat], "iscrowd": 0})
            ann_id += 1

    base_scenes = []
    for s in range(n_scenes):
        arr, items = generate_scene(size, rng, classes, probs, max_items, noise_std)
        base_scenes.append((arr, items))
        _add(arr, items, scene_id=s)
    n_dup = int(round(duplicate_fraction * n_scenes))
    for s in rng.choice(n_scenes, size=n_dup, replace=False) if n_dup else []:
        arr, items = base_scenes[int(s)]
        _add(_near_duplicate(arr, rng), items, scene_id=int(s))

    out = {"info": {"description": "Synthetic TACO-like waste scenes"}, "images": images,
           "annotations": anns, "categories": cats}
    json_path = root / "annotations.json"
    json_path.write_text(json.dumps(out), encoding="utf-8")
    logger.info("Wrote %d synthetic scenes (%d near-duplicates), %d annotations -> %s",
                len(images), n_dup, len(anns), json_path)
    return json_path


def write_synthetic_trashnet(root: Path | str, n_per_class: int = 60, size: int = 128, seed: int = 42,
                             classes: Sequence[str] = DEFAULT_CLASSES, duplicate_fraction: float = 0.05,
                             noise_std: float = 6.0, rare_classes: Sequence[str] = ("trash",),
                             rare_weight: float = 0.5) -> Path:
    """Write a TrashNet-like folder dataset ``root/<class>/<class>_<k>.png``; rare classes get fewer samples."""
    set_seed(seed)
    rng = np.random.default_rng(seed)
    root = ensure_dir(root)
    for c in classes:
        cdir = ensure_dir(root / c)
        n = max(2, int(n_per_class * (rare_weight if c in rare_classes else 1.0)))
        for k in range(n):
            arr = render_single_item(c, size, rng, noise_std)
            Image.fromarray(arr).save(cdir / f"{c}_{k:04d}.png")
            if rng.random() < duplicate_fraction:
                Image.fromarray(_near_duplicate(arr, rng)).save(cdir / f"{c}_{k:04d}_dup.png")
    logger.info("Wrote synthetic TrashNet-style dataset -> %s", root)
    return root


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        p = write_synthetic_coco(Path(tmp) / "coco", n_scenes=6)
        print("COCO JSON:", p)
