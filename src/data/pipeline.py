"""End-to-end dataset governance pipeline: acquire -> dedup -> grouped split -> convert -> report.

Usage (real data is the default; synthetic is opt-in for tests only):
    python -m src.data.pipeline --source taco --download
    python -m src.data.pipeline --source trashnet --download
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from src.data import converters
from src.data.coco import DetectionSample
from src.data.dedup import find_near_duplicates
from src.data.loaders import DataNotFoundError
from src.data.splits import assert_no_leakage, grouped_split, leakage_report
from src.data.synthetic import write_synthetic_coco
from src.data.taco import download_taco, load_taco
from src.data.trashnet import download_trashnet, scan_trashnet
from src.utils import (ensure_dir, get_logger, load_config, resolve_path, save_figure, save_json, set_seed)

logger = get_logger(__name__)


def _plot_class_distribution(counts: Dict[str, Dict[str, int]], class_names: List[str], title: str, name: str,
                             cfg: Dict[str, Any]) -> Path:
    fig, ax = plt.subplots(figsize=(7, 3.8))
    splits = list(counts)
    w = 0.8 / len(splits)
    x = np.arange(len(class_names))
    for k, s in enumerate(splits):
        ax.bar(x + k * w, [counts[s].get(c, 0) for c in class_names], w, label=s)
    ax.set_xticks(x + w * (len(splits) - 1) / 2)
    ax.set_xticklabels(class_names, rotation=30)
    ax.set_ylabel("instances")
    ax.set_title(title)
    ax.legend()
    return save_figure(fig, name, subdir="data", cfg=cfg)


def _detection_pipeline(samples: List[DetectionSample], cfg: Dict[str, Any], tag: str) -> Dict[str, Any]:
    names = cfg["classes"]["names"]
    seed = cfg["project"]["seed"]
    dd = cfg["data"]["dedup"]
    keep_idx = list(range(len(samples)))
    cluster = list(range(len(samples)))
    if dd["enabled"]:
        res = find_near_duplicates([s.image_path for s in samples], dd["max_hamming_distance"], dd["hash_size"])
        keep_idx, cluster = res.kept, res.cluster_of
    kept = [samples[i] for i in keep_idx]
    groups = [(f"scene{s.scene_id}" if s.scene_id is not None else f"cluster{cluster[i]}")
              for i, s in zip(keep_idx, kept)]
    labels = [int(s.labels[0]) if len(s.labels) else 0 for s in kept]
    splits = grouped_split(groups, cfg["data"]["split"]["ratios"], seed, labels)
    assert_no_leakage(groups, splits)

    out = ensure_dir(Path(cfg["paths"]["processed_root"]) / tag)
    converters.to_yolo(kept, splits, out / "yolo", names)
    converters.to_coco_splits(kept, splits, out / "coco", names)
    converters.to_semantic_masks(kept, splits, out / "semantic")
    converters.to_classification_crops(kept, splits, out / "crops", names, cfg["data"]["crop_padding"])

    counts = {sp: {names[c]: int(sum((kept[i].labels == c).sum() for i in idxs)) for c in range(len(names))}
              for sp, idxs in splits.items()}
    fig = _plot_class_distribution(counts, names, f"{tag}: instances per class / split", f"{tag}_class_distribution", cfg)
    summary = {"source": tag, "n_images_raw": len(samples), "n_images_after_dedup": len(kept),
               "leakage": leakage_report(groups, splits), "class_counts": counts, "figure": str(fig),
               "processed_dir": str(out)}
    save_json(summary, f"{tag}_dataset_summary", cfg=cfg)
    return summary


def _classification_pipeline(cfg: Dict[str, Any], root: Path, tag: str) -> Dict[str, Any]:
    names = cfg["classes"]["names"]
    paths, labels = scan_trashnet(root, names)
    if not paths:
        raise FileNotFoundError(f"No TrashNet images found in {root}")
    dd = cfg["data"]["dedup"]
    keep = list(range(len(paths)))
    cluster = keep
    if dd["enabled"]:
        res = find_near_duplicates(paths, dd["max_hamming_distance"], dd["hash_size"])
        keep, cluster = res.kept, res.cluster_of
    kp, kl = [paths[i] for i in keep], [labels[i] for i in keep]
    groups = [f"cluster{cluster[i]}" for i in keep]
    splits = grouped_split(groups, cfg["data"]["split"]["ratios"], cfg["project"]["seed"], kl)
    assert_no_leakage(groups, splits)
    out = ensure_dir(Path(cfg["paths"]["processed_root"]) / tag / "classification")
    converters.to_folder_classification(kp, kl, splits, out, names)
    counts = {sp: {names[c]: int(sum(1 for i in idxs if kl[i] == c)) for c in range(len(names))}
              for sp, idxs in splits.items()}
    fig = _plot_class_distribution(counts, names, f"{tag}: images per class / split", f"{tag}_class_distribution", cfg)
    summary = {"source": tag, "n_images_raw": len(paths), "n_images_after_dedup": len(kp),
               "leakage": leakage_report(groups, splits), "class_counts": counts, "figure": str(fig),
               "processed_dir": str(out)}
    save_json(summary, f"{tag}_dataset_summary", cfg=cfg)
    return summary


def prepare_dataset(cfg: Optional[Dict[str, Any]] = None, source: str = "trashnet", download: bool = False,
                    max_images: Optional[int] = None) -> Dict[str, Any]:
    """Run the governance pipeline for ``trashnet`` | ``taco`` (real data) or ``synthetic`` (explicit opt-in).

    Real sources never fall back to synthetic data: if the dataset is missing and cannot be downloaded a
    :class:`DataNotFoundError` with instructions is raised.
    """
    cfg = cfg or load_config()
    set_seed(cfg["project"]["seed"])
    p, d, syn = cfg["paths"], cfg["data"], cfg["data"]["synthetic"]
    names, rare = cfg["classes"]["names"], cfg["classes"]["rare"]

    if source == "taco":
        raw = resolve_path(p["raw_taco"])
        ann = raw / "annotations.json"
        if download and not ann.exists():
            download_taco(raw, d["urls"]["taco_annotations"], max_images)
        elif download:  # annotations present -> (re)try missing images
            download_taco(raw, d["urls"]["taco_annotations"], max_images)
        if not ann.exists():
            raise DataNotFoundError(f"TACO annotations not found at {ann}")
        samples, stats = load_taco(ann, raw, names, cfg["taco_keyword_map"], require_exists=True)
        logger.info("TACO parse stats: %s", stats)
        if not samples:
            raise DataNotFoundError(f"TACO annotations found but no image files present in {raw}")
        return _detection_pipeline(samples, cfg, "taco")

    if source == "trashnet":
        raw = resolve_path(p["raw_trashnet"])
        if download and not any(raw.glob("**/*.jpg")):
            download_trashnet(raw, d["urls"]["trashnet_zip"])
        if not any(raw.glob("**/*.jpg")) and not any(raw.glob("**/*.png")):
            raise DataNotFoundError(f"No TrashNet images found in {raw}")
        return _classification_pipeline(cfg, raw, "trashnet")

    if source == "synthetic":  # explicit opt-in (tests / demos only)
        syn_root = resolve_path(p["synthetic_root"]) / "taco"
        json_path = write_synthetic_coco(syn_root, syn["n_scenes"], d["image_size"], cfg["project"]["seed"], names,
                                         syn["max_items_per_scene"], syn["duplicate_fraction"], syn["noise_std"],
                                         rare, syn["rare_class_weight"])
        samples, _ = load_taco(json_path, syn_root / "images", names, cfg["taco_keyword_map"])
        return _detection_pipeline(samples, cfg, "synthetic_taco")

    raise ValueError(f"unknown source '{source}'")


def main() -> None:
    ap = argparse.ArgumentParser(description="Dataset governance pipeline")
    ap.add_argument("--source", choices=["synthetic", "taco", "trashnet"], default="trashnet")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    s = prepare_dataset(load_config(a.config), a.source, a.download, a.max_images)
    print({k: v for k, v in s.items() if k != "class_counts"})


if __name__ == "__main__":
    main()
