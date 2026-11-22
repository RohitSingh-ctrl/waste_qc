"""Label converters: unified samples -> YOLO, COCO(split), semantic masks, classification crops."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import yaml
from PIL import Image

from src.data.coco import DetectionSample, polygons_to_mask
from src.utils import ensure_dir

Splits = Mapping[str, Sequence[int]]  # split name -> indices into the samples list


def _name(s: DetectionSample) -> str:
    return f"{s.image_id:06d}"


def to_yolo(samples: Sequence[DetectionSample], splits: Splits, out_dir: Path | str,
            class_names: Sequence[str]) -> Path:
    """Write a YOLOv8 dataset (``images/<split>``, ``labels/<split>``, ``data.yaml``).

    Labels are ``cls cx cy w h`` normalised to [0, 1]. Returns the ``data.yaml`` path.
    """
    out = ensure_dir(out_dir)
    for split, idxs in splits.items():
        ensure_dir(out / "images" / split)
        ensure_dir(out / "labels" / split)
        for i in idxs:
            s = samples[i]
            shutil.copy2(s.image_path, out / "images" / split / f"{_name(s)}{s.image_path.suffix}")
            lines: List[str] = []
            for (x0, y0, x1, y1), c in zip(s.boxes, s.labels):
                cx, cy = (x0 + x1) / 2 / s.width, (y0 + y1) / 2 / s.height
                w, h = (x1 - x0) / s.width, (y1 - y0) / s.height
                lines.append(f"{int(c)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            (out / "labels" / split / f"{_name(s)}.txt").write_text("\n".join(lines), encoding="utf-8")
    data_yaml = out / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({
        "path": str(out.resolve()), "train": "images/train", "val": "images/val", "test": "images/test",
        "names": {i: n for i, n in enumerate(class_names)}}), encoding="utf-8")
    return data_yaml


def to_coco_splits(samples: Sequence[DetectionSample], splits: Splits, out_dir: Path | str,
                   class_names: Sequence[str]) -> Dict[str, Path]:
    """Write one COCO ``instances_<split>.json`` per split (unified categories, polygons kept)."""
    out = ensure_dir(out_dir)
    cats = [{"id": i + 1, "name": n, "supercategory": n} for i, n in enumerate(class_names)]
    paths: Dict[str, Path] = {}
    for split, idxs in splits.items():
        images, anns, aid = [], [], 1
        for i in idxs:
            s = samples[i]
            images.append({"id": s.image_id, "file_name": str(s.image_path), "width": s.width, "height": s.height})
            for (x0, y0, x1, y1), c, poly in zip(s.boxes, s.labels, s.polygons):
                anns.append({"id": aid, "image_id": s.image_id, "category_id": int(c) + 1,
                             "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
                             "area": float((x1 - x0) * (y1 - y0)), "segmentation": poly, "iscrowd": 0})
                aid += 1
        p = out / f"instances_{split}.json"
        p.write_text(json.dumps({"images": images, "annotations": anns, "categories": cats}), encoding="utf-8")
        paths[split] = p
    return paths


def sample_to_semantic_mask(s: DetectionSample) -> np.ndarray:
    """Semantic mask ``uint8 (H, W)``: 0 = background, ``class_idx + 1`` otherwise (small objects painted last)."""
    mask = np.zeros((s.height, s.width), dtype=np.uint8)
    areas = [(float((b[2] - b[0]) * (b[3] - b[1])), k) for k, b in enumerate(s.boxes)]
    for _, k in sorted(areas, reverse=True):
        m = polygons_to_mask(s.polygons[k], s.width, s.height) if s.polygons[k] else _box_mask(s, k)
        mask[m] = int(s.labels[k]) + 1
    return mask


def _box_mask(s: DetectionSample, k: int) -> np.ndarray:
    m = np.zeros((s.height, s.width), dtype=bool)
    x0, y0, x1, y1 = [int(round(v)) for v in s.boxes[k]]
    m[max(0, y0):y1, max(0, x0):x1] = True
    return m


def to_semantic_masks(samples: Sequence[DetectionSample], splits: Splits, out_dir: Path | str) -> Path:
    """Write ``images/<split>`` + ``masks/<split>`` (PNG, 0=background) for U-Net training."""
    out = ensure_dir(out_dir)
    for split, idxs in splits.items():
        ensure_dir(out / "images" / split)
        ensure_dir(out / "masks" / split)
        for i in idxs:
            s = samples[i]
            shutil.copy2(s.image_path, out / "images" / split / f"{_name(s)}{s.image_path.suffix}")
            Image.fromarray(sample_to_semantic_mask(s)).save(out / "masks" / split / f"{_name(s)}.png")
    return out


def to_classification_crops(samples: Sequence[DetectionSample], splits: Splits, out_dir: Path | str,
                            class_names: Sequence[str], padding: float = 0.10, min_size: int = 8) -> Path:
    """Crop each annotated item into ``<split>/<class>/<img>_<k>.png`` (with relative padding)."""
    out = ensure_dir(out_dir)
    for split, idxs in splits.items():
        for c in class_names:
            ensure_dir(out / split / c)
        for i in idxs:
            s = samples[i]
            with Image.open(s.image_path) as im:
                im = im.convert("RGB")
                for k, ((x0, y0, x1, y1), c) in enumerate(zip(s.boxes, s.labels)):
                    pw, ph = (x1 - x0) * padding, (y1 - y0) * padding
                    box = (max(0, int(x0 - pw)), max(0, int(y0 - ph)),
                           min(s.width, int(np.ceil(x1 + pw))), min(s.height, int(np.ceil(y1 + ph))))
                    if box[2] - box[0] < min_size or box[3] - box[1] < min_size:
                        continue
                    im.crop(box).save(out / split / class_names[int(c)] / f"{_name(s)}_{k}.png")
    return out


def to_folder_classification(paths: Sequence[Path], labels: Sequence[int], splits: Splits, out_dir: Path | str,
                             class_names: Sequence[str]) -> Path:
    """Copy a TrashNet-style list into ``<split>/<class>/`` folders."""
    out = ensure_dir(out_dir)
    for split, idxs in splits.items():
        for c in class_names:
            ensure_dir(out / split / c)
        for i in idxs:
            shutil.copy2(paths[i], out / split / class_names[labels[i]] / Path(paths[i]).name)
    return out
