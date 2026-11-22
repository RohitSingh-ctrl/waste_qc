"""Robustness stress tests (Industry readiness): how much does macro-F1 drop under realistic belt conditions?

Stresses: sensor noise, blur (motion/focus), dark / bright lighting, JPEG compression, partial occlusion.
``relative_drop = (clean_f1 - stressed_f1) / clean_f1`` is compared with ``deployment.max_robustness_drop`` in the config.
"""
from __future__ import annotations

import argparse
import io
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from src.evaluation.metrics import classification_report
from src.utils import get_logger, load_config, save_figure, save_json
from src.vision.common import get_device

logger = get_logger(__name__)


def _noise(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    a = np.asarray(img, np.float32) + rng.normal(0, 25, (img.size[1], img.size[0], 3))
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def _jpeg(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=20)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _occlude(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    out = img.copy()
    w, h = out.size
    bw, bh = w // 2, h // 2                       # a box covering 25% of the image
    x0, y0 = int(rng.integers(0, w - bw + 1)), int(rng.integers(0, h - bh + 1))
    ImageDraw.Draw(out).rectangle([x0, y0, x0 + bw, y0 + bh], fill=(0, 0, 0))
    return out


STRESSES: Dict[str, Callable[[Image.Image, np.random.Generator], Image.Image]] = {
    "gaussian_noise": _noise,
    "blur": lambda im, rng: im.filter(ImageFilter.GaussianBlur(radius=max(1.5, im.size[0] / 64))),
    "low_brightness": lambda im, rng: ImageEnhance.Brightness(im).enhance(0.4),
    "high_brightness": lambda im, rng: ImageEnhance.Brightness(im).enhance(1.8),
    "jpeg_q20": _jpeg,
    "occlusion_25pct": _occlude,
}


def apply_stress(img: Image.Image, name: str, seed: int = 0) -> Image.Image:
    """Apply one named stress to an RGB PIL image (same size back)."""
    return STRESSES[name](img.convert("RGB"), np.random.default_rng(seed))


@torch.no_grad()
def _predict(model: torch.nn.Module, imgs: Sequence[Image.Image], size: int, device: torch.device, batch_size: int) -> np.ndarray:
    from src.vision.datasets import classification_transforms
    tf = classification_transforms(size, train=False)
    preds: List[np.ndarray] = []
    for s in range(0, len(imgs), batch_size):
        x = torch.stack([tf(i.convert("RGB")) for i in imgs[s:s + batch_size]]).to(device)
        preds.append(model(x).argmax(1).cpu().numpy())
    return np.concatenate(preds)


def evaluate_robustness_on(model: torch.nn.Module, items: Sequence[Tuple[Image.Image, int]], names: Sequence[str],
                           size: int, device: Optional[torch.device] = None, stresses: Optional[Sequence[str]] = None,
                           batch_size: int = 32, seed: int = 42) -> Dict[str, Any]:
    """Clean vs stressed macro-F1/accuracy for a classifier and a list of ``(PIL image, label)``."""
    device = device or get_device()
    model = model.to(device).eval()
    imgs, y = [i for i, _ in items], np.array([l for _, l in items])
    clean = classification_report(y, _predict(model, imgs, size, device, batch_size), names)
    out: Dict[str, Any] = {"n_images": len(items), "clean": {"accuracy": clean["accuracy"], "macro_f1": clean["macro_f1"]},
                           "stress": {}}
    for s in (stresses or list(STRESSES)):
        rep = classification_report(y, _predict(model, [apply_stress(i, s, seed + k) for k, i in enumerate(imgs)], size,
                                                device, batch_size), names)
        base = clean["macro_f1"]
        out["stress"][s] = {"accuracy": rep["accuracy"], "macro_f1": rep["macro_f1"],
                            "relative_drop": float((base - rep["macro_f1"]) / base) if base > 0 else 0.0}
    out["max_relative_drop"] = max(v["relative_drop"] for v in out["stress"].values())
    out["worst_stress"] = max(out["stress"], key=lambda k: out["stress"][k]["relative_drop"])
    return out


def run_robustness(model_name: str, cfg: Optional[Dict[str, Any]] = None, image_size: int = 224, synthetic: bool = False,
                   max_images: Optional[int] = 300) -> Dict[str, Any]:
    """Load ``outputs/checkpoints/<model>.pt`` and stress-test it on the processed test split."""
    from src.vision.classification import build_classifier
    from src.vision.common import checkpoint_path, classification_root
    from src.vision.datasets import FolderClassification

    cfg = cfg or load_config()
    names = cfg["classes"]["names"]
    ck_path = checkpoint_path(cfg, model_name)
    if not ck_path.exists():
        raise FileNotFoundError(f"{ck_path} not found. Train first: python -m src.vision.classification --model {model_name}")
    ck = torch.load(ck_path, map_location="cpu")
    model = build_classifier(model_name, len(names), pretrained=False)
    model.load_state_dict(ck["state_dict"])
    ds = FolderClassification(classification_root(cfg, synthetic), "test", names, image_size, augment=False)
    idx = np.random.default_rng(cfg["project"]["seed"]).permutation(len(ds.items))[:max_images]
    items = [(Image.open(ds.items[i][0]).convert("RGB"), ds.items[i][1]) for i in idx]
    res = evaluate_robustness_on(model, items, names, image_size)
    res["model"] = model_name
    fig, ax = plt.subplots(figsize=(8, 3.6))
    keys = list(res["stress"])
    ax.bar(keys, [res["stress"][k]["relative_drop"] for k in keys])
    ax.axhline(cfg["deployment"]["max_robustness_drop"], color="r", ls="--", label="allowed drop")
    ax.set(title=f"{model_name}: relative macro-F1 drop under stress", ylabel="relative drop")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.legend()
    save_figure(fig, f"robustness_{model_name}", "vision", cfg=cfg)
    save_json(res, f"robustness_{model_name}", "vision", cfg=cfg)
    logger.info("%s clean F1=%.3f, worst stress=%s (drop %.1f%%)", model_name, res["clean"]["macro_f1"], res["worst_stress"],
                100 * res["max_relative_drop"])
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="Stress-test a trained classifier")
    ap.add_argument("--model", choices=["resnet50", "googlenet", "all"], default="all")
    ap.add_argument("--image-size", type=int, default=224, help="must match the size used in training")
    ap.add_argument("--max-images", type=int, default=300)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    for m in (cfg["vision"]["classification"]["models"] if a.model == "all" else [a.model]):
        run_robustness(m, cfg, a.image_size, a.synthetic, a.max_images)


if __name__ == "__main__":
    main()
