"""Task metrics (NumPy only): classification, recycling purity, segmentation IoU, detection mAP."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from src.utils import save_figure


# ----------------------------------------------------------------------------- classification
def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int], n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def classification_report(y_true: Sequence[int], y_pred: Sequence[int], names: Sequence[str]) -> Dict[str, Any]:
    """Accuracy, macro-F1 and per-class precision/recall/F1 (classes with no samples are skipped in macro)."""
    cm = confusion_matrix(y_true, y_pred, len(names))
    tp = np.diag(cm).astype(float)
    prec = np.divide(tp, cm.sum(0), out=np.zeros_like(tp), where=cm.sum(0) > 0)
    rec = np.divide(tp, cm.sum(1), out=np.zeros_like(tp), where=cm.sum(1) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    present = cm.sum(1) > 0
    return {"accuracy": float(tp.sum() / max(cm.sum(), 1)), "macro_f1": float(f1[present].mean()) if present.any() else 0.0,
            "per_class": {n: {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i]),
                              "support": int(cm[i].sum())} for i, n in enumerate(names)},
            "confusion_matrix": cm.tolist()}


def plot_confusion_matrix(cm: np.ndarray, names: Sequence[str], name: str, subdir: str = "vision") -> str:
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set(xticks=range(len(names)), yticks=range(len(names)), xticklabels=names, yticklabels=names,
           xlabel="predicted", ylabel="true", title=name)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    return str(save_figure(fig, name, subdir))


def recycling_purity(y_true: Sequence[int], y_pred: Sequence[int], names: Sequence[str],
                     recyclable: Sequence[str]) -> Dict[str, float]:
    """Industry metric. Items the system sends to the *recyclable stream* (pred in ``recyclable``):

    * ``purity_rate`` = fraction of them that are truly recyclable (target >= 0.95 in config)
    * ``contamination_rate`` = 1 - purity (e.g. a 'trash' item sorted as plastic)
    * ``recovery_rate`` = fraction of truly recyclable items that reach the recyclable stream
    """
    rec_ids = np.array([names.index(c) for c in recyclable])
    t, p = np.asarray(y_true), np.asarray(y_pred)
    sent, truly = np.isin(p, rec_ids), np.isin(t, rec_ids)
    purity = float((sent & truly).sum() / sent.sum()) if sent.any() else 0.0
    recovery = float((sent & truly).sum() / truly.sum()) if truly.any() else 0.0
    return {"purity_rate": purity, "contamination_rate": 1.0 - purity if sent.any() else 0.0, "recovery_rate": recovery}


# ----------------------------------------------------------------------------- segmentation
def segmentation_metrics(pred: np.ndarray, target: np.ndarray, n_classes: int) -> Dict[str, Any]:
    """Per-class IoU / Dice and means over classes present in prediction or target. Inputs: int arrays, same shape."""
    cm = confusion_matrix(target.ravel(), pred.ravel(), n_classes).astype(float)
    tp, fp, fn = np.diag(cm), cm.sum(0) - np.diag(cm), cm.sum(1) - np.diag(cm)
    denom_iou, denom_dice = tp + fp + fn, 2 * tp + fp + fn
    iou = np.where(denom_iou > 0, tp / np.maximum(denom_iou, 1), np.nan)
    dice = np.where(denom_dice > 0, 2 * tp / np.maximum(denom_dice, 1), np.nan)
    return {"pixel_acc": float(tp.sum() / max(cm.sum(), 1)), "miou": float(np.nanmean(iou)) if np.isfinite(iou).any() else 0.0,
            "mean_dice": float(np.nanmean(dice)) if np.isfinite(dice).any() else 0.0,
            "per_class_iou": [None if np.isnan(v) else float(v) for v in iou]}


# ----------------------------------------------------------------------------- detection
def box_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU of ``(N,4)`` and ``(M,4)`` xyxy boxes -> ``(N,M)``."""
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    lt, rb = np.maximum(a[:, None, :2], b[None, :, :2]), np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area_a, area_b = (a[:, 2:] - a[:, :2]).prod(-1), (b[:, 2:] - b[:, :2]).prod(-1)
    return inter / np.maximum(area_a[:, None] + area_b[None] - inter, 1e-9)


def _ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """Area under the monotone precision envelope (all-point interpolation, COCO/VOC2010 style)."""
    r, p = np.concatenate([[0.0], recall, [1.0]]), np.concatenate([[0.0], precision, [0.0]])
    p = np.maximum.accumulate(p[::-1])[::-1]
    idx = np.where(r[1:] != r[:-1])[0]
    return float(((r[idx + 1] - r[idx]) * p[idx + 1]).sum())


def detection_map(preds: List[Dict[str, np.ndarray]], gts: List[Dict[str, np.ndarray]], n_classes: int,
                  iou_thresholds: Optional[Sequence[float]] = None) -> Dict[str, Any]:
    """mAP@0.5 and mAP@0.5:0.95 (10 thresholds).

    ``preds[i]`` = ``{boxes (K,4), scores (K,), labels (K,)}``; ``gts[i]`` = ``{boxes, labels}``; labels are 0-based.
    Classes without ground truth are excluded from the mean.
    """
    ths = list(iou_thresholds) if iou_thresholds is not None else [round(0.5 + 0.05 * k, 2) for k in range(10)]
    ap = np.full((len(ths), n_classes), np.nan)
    for c in range(n_classes):
        n_gt = sum(int((g["labels"] == c).sum()) for g in gts)
        if n_gt == 0:
            continue
        dets = [(float(s), i, b) for i, p in enumerate(preds) for b, s, l in zip(p["boxes"], p["scores"], p["labels"]) if l == c]
        dets.sort(key=lambda d: -d[0])
        for ti, th in enumerate(ths):
            used = {i: np.zeros(int((g["labels"] == c).sum()), bool) for i, g in enumerate(gts)}
            tp = np.zeros(len(dets))
            for k, (_, i, box) in enumerate(dets):
                gb = gts[i]["boxes"][gts[i]["labels"] == c]
                if len(gb) == 0:
                    continue
                ious = box_iou_matrix(box[None], gb)[0]
                j = int(ious.argmax())
                if ious[j] >= th and not used[i][j]:
                    used[i][j], tp[k] = True, 1
            ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
            ap[ti, c] = _ap(ctp / n_gt, ctp / np.maximum(ctp + cfp, 1e-9)) if len(dets) else 0.0
    valid = ~np.isnan(ap[0])
    return {"map50": float(np.nanmean(ap[0])) if valid.any() else 0.0,
            "map50_95": float(np.nanmean(ap[:, valid])) if valid.any() else 0.0,
            "ap50_per_class": [None if np.isnan(v) else float(v) for v in ap[0]]}
