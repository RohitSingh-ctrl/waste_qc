"""Inference helpers for the GUI (no Streamlit imports, so they can be unit-tested)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image

from src.utils import load_config
from src.vision.common import checkpoint_path, get_device

CLASSIFIERS = ("resnet50", "googlenet")


def list_classifiers(cfg: Optional[Dict[str, Any]] = None) -> List[str]:
    """Classifier names that have a trained checkpoint in ``paths.checkpoints``."""
    cfg = cfg or load_config()
    return [m for m in CLASSIFIERS if checkpoint_path(cfg, m).exists()]


def load_classifier(name: str, cfg: Optional[Dict[str, Any]] = None) -> Tuple[torch.nn.Module, List[str]]:
    """Load a trained ResNet-50 / GoogLeNet checkpoint written by ``src.vision.classification``."""
    from src.vision.classification import build_classifier
    cfg = cfg or load_config()
    path = checkpoint_path(cfg, name)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Train first: python -m src.vision.classification --model {name}")
    ck = torch.load(path, map_location="cpu")
    names = list(ck.get("classes") or cfg["classes"]["names"])
    model = build_classifier(name, len(names), pretrained=False)
    model.load_state_dict(ck["state_dict"])
    return model.to(get_device()).eval(), names


@torch.no_grad()
def predict_image(model: torch.nn.Module, img: Image.Image, names: Sequence[str], size: int = 224) -> Dict[str, float]:
    """Softmax probabilities per class for one PIL image (``size`` must match the training image size)."""
    from src.vision.datasets import classification_transforms
    x = classification_transforms(size, train=False)(img.convert("RGB")).unsqueeze(0).to(next(model.parameters()).device)
    p = torch.softmax(model(x), 1)[0].cpu().numpy()
    return {n: float(v) for n, v in zip(names, p)}


def route_item(probs: Dict[str, float], cfg: Dict[str, Any], min_confidence: float = 0.5) -> Dict[str, Any]:
    """Sorting decision: recycle stream, reject stream, or manual check when the model is unsure."""
    label = max(probs, key=probs.get)
    conf = probs[label]
    if conf < min_confidence:
        action = "manual check"
    else:
        action = "recycle" if label in cfg["classes"]["recyclable"] else "reject"
    return {"label": label, "confidence": conf, "action": action}


def batch_summary(decisions: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Composition of a processed batch. Rates are *predicted* (no ground truth in the GUI)."""
    n = len(decisions)
    counts: Dict[str, int] = {}
    for d in decisions:
        counts[d["label"]] = counts.get(d["label"], 0) + 1
    actions = [d["action"] for d in decisions]
    return {"n": n, "class_counts": counts,
            "recycle": actions.count("recycle"), "reject": actions.count("reject"), "manual_check": actions.count("manual check"),
            "predicted_reject_share": actions.count("reject") / n if n else 0.0}
