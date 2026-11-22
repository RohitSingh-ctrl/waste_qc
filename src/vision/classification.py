"""Unit IV (a): image classification with ResNet-50 and GoogLeNet (torchvision, trained from scratch by default)."""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

from src.evaluation.benchmark import count_parameters, measure_latency, model_size_mb
from src.evaluation.metrics import classification_report, plot_confusion_matrix, recycling_purity
from src.utils import get_logger, load_config, save_figure, save_json, set_seed
from src.vision.common import (checkpoint_path, classification_root, get_device, parse_common_args, Timer)
from src.vision.datasets import FolderClassification, class_weights

import matplotlib.pyplot as plt

logger = get_logger(__name__)


def build_classifier(name: str, n_classes: int, pretrained: bool = False) -> nn.Module:
    """ResNet-50 or GoogLeNet with the head replaced for ``n_classes`` (aux heads disabled for GoogLeNet)."""
    if name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, n_classes)
    elif name == "googlenet":
        m = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT if pretrained else None, aux_logits=pretrained,
                             init_weights=True)
        m.aux_logits, m.aux1, m.aux2 = False, None, None
        m.fc = nn.Linear(m.fc.in_features, n_classes)
    else:
        raise ValueError(f"unknown classifier '{name}'")
    return m


@torch.no_grad()
def predict_all(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        out = model(x.to(device))
        ps.append(out.argmax(1).cpu().numpy())
        ys.append(np.asarray(y))
    return np.concatenate(ys), np.concatenate(ps)


def train_classifier(name: str, cfg: Optional[Dict[str, Any]] = None, synthetic: bool = False,
                     epochs: Optional[int] = None, image_size: int = 224) -> Dict[str, Any]:
    """Train + evaluate one classifier on the processed folders; saves curves, confusion matrix, JSON, checkpoint."""
    cfg = cfg or load_config()
    vc = cfg["vision"]["classification"]
    set_seed(cfg["project"]["seed"])
    names, device = cfg["classes"]["names"], get_device()
    root = classification_root(cfg, synthetic)
    tr = FolderClassification(root, "train", names, image_size, augment=True)
    va = FolderClassification(root, "val", names, image_size, augment=False)
    te = FolderClassification(root, "test", names, image_size, augment=False)
    bs = vc["batch_size"]
    dl = {"train": DataLoader(tr, bs, shuffle=True), "val": DataLoader(va, bs), "test": DataLoader(te, bs)}

    model = build_classifier(name, len(names), vc["pretrained"]).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(tr.targets, len(names)).to(device))
    opt = torch.optim.Adam(model.parameters(), lr=vc["lr"])
    epochs = epochs or vc["epochs"]
    hist = {"train_loss": [], "val_acc": []}
    logger.info("%s on %s | device=%s | train/val/test=%d/%d/%d", name, root, device, len(tr), len(va), len(te))
    with Timer() as t:
        for ep in range(1, epochs + 1):
            model.train()
            run = 0.0
            for x, y in dl["train"]:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                loss = loss_fn(model(x), y)
                loss.backward()
                opt.step()
                run += loss.item() * len(y)
            yv, pv = predict_all(model, dl["val"], device)
            hist["train_loss"].append(run / len(tr))
            hist["val_acc"].append(float((yv == pv).mean()))
            logger.info("epoch %d/%d loss=%.4f val_acc=%.3f", ep, epochs, hist["train_loss"][-1], hist["val_acc"][-1])
    yt, pt = predict_all(model, dl["test"], device)
    rep = classification_report(yt, pt, names)
    rep.update(recycling_purity(yt, pt, names, cfg["classes"]["recyclable"]))
    sample = next(iter(dl["test"]))[0][:1].to(device)
    model.eval()
    with torch.no_grad():
        lat = measure_latency(lambda: model(sample), warmup=2, runs=10)
    result = {"model": name, "epochs": epochs, "train_seconds": t.seconds, "params": count_parameters(model),
              "size_mb": model_size_mb(model), **lat, "history": hist, "test": rep}
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    ax[0].plot(hist["train_loss"]); ax[0].set(title=f"{name} train loss", xlabel="epoch")
    ax[1].plot(hist["val_acc"]); ax[1].set(title=f"{name} val accuracy", xlabel="epoch")
    save_figure(fig, f"{name}_curves", "vision")
    plot_confusion_matrix(np.array(rep["confusion_matrix"]), names, f"{name}_confusion", "vision")
    torch.save({"model": name, "state_dict": model.state_dict(), "classes": names}, checkpoint_path(cfg, name))
    save_json(result, f"classification_{name}", "vision")
    logger.info("%s test acc=%.3f macro_f1=%.3f purity=%.3f", name, rep["accuracy"], rep["macro_f1"], rep["purity_rate"])
    return result


def main() -> None:
    ap = parse_common_args("Train ResNet-50 / GoogLeNet classifiers")
    ap.add_argument("--model", choices=["resnet50", "googlenet", "all"], default="all")
    a = ap.parse_args()
    cfg = load_config(a.config)
    for m in (cfg["vision"]["classification"]["models"] if a.model == "all" else [a.model]):
        train_classifier(m, cfg, a.synthetic, a.epochs, a.image_size or 224)


if __name__ == "__main__":
    main()
