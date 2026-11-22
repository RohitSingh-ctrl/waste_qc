"""Unit III: controlled ablations (initialization, normalization, regularization, augmentation).

One factor is varied at a time against a fixed baseline; data, split, seed, epochs, optimizer and batch order are
identical across runs so differences are attributable to the factor. Small CNN on ``size x size`` images.
"""
from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.loaders import load_trashnet_arrays
from src.data.synthetic import make_classification_arrays
from src.evaluation.metrics import classification_report
from src.utils import get_logger, load_config, save_figure, save_json, set_seed

logger = get_logger(__name__)
BASELINE = {"init": "he", "batch_norm": False, "dropout": 0.0, "weight_decay": 0.0, "augmentation": False}


def make_cnn(n_classes: int, init: str, batch_norm: bool, dropout: float, size: int = 32) -> nn.Module:
    def block(cin: int, cout: int) -> List[nn.Module]:
        return [nn.Conv2d(cin, cout, 3, padding=1), *([nn.BatchNorm2d(cout)] if batch_norm else []), nn.ReLU(), nn.MaxPool2d(2)]

    net = nn.Sequential(*block(3, 16), *block(16, 32), *block(32, 64), nn.Flatten(), nn.Dropout(dropout),
                        nn.Linear(64 * (size // 8) ** 2, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, n_classes))
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            if init == "random":
                nn.init.normal_(m.weight, 0.0, 0.01)
            elif init == "xavier":
                nn.init.xavier_normal_(m.weight)
            elif init == "he":
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            else:
                raise ValueError(init)
            nn.init.zeros_(m.bias)
    return net


def augment_batch(x: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """Random horizontal flip + up to 3px circular shift (tensor-level, fast on CPU/GPU)."""
    flip = torch.rand(len(x), generator=g) < 0.5
    x = torch.where(flip[:, None, None, None].to(x.device), x.flip(-1), x)
    dx, dy = [int(v) for v in torch.randint(-3, 4, (2,), generator=g)]
    return torch.roll(x, shifts=(dy, dx), dims=(2, 3))


def _to_tensor(imgs: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(imgs).permute(0, 3, 1, 2).float() / 255.0
    return (x - 0.5) / 0.25


def load_splits(cfg: Dict[str, Any], size: int, synthetic: bool, per_class: Optional[int] = None):
    """Returns ``{train,val,test: (X, y)}``. Synthetic mode uses different seeds per split (no leakage)."""
    names, seed = cfg["classes"]["names"], cfg["project"]["seed"]
    out = {}
    for k, s in enumerate(("train", "val", "test")):
        if synthetic:
            imgs, y = make_classification_arrays((per_class or 40) if s == "train" else 15, size, names, seed + 100 * k)
        else:
            imgs, y = load_trashnet_arrays(cfg, size, s, per_class, False, seed)
        out[s] = (_to_tensor(imgs), torch.from_numpy(np.asarray(y)).long())
    return out


def run_one(factors: Dict[str, Any], data: Dict[str, Tuple[torch.Tensor, torch.Tensor]], n_classes: int, epochs: int,
            seed: int, size: int, batch_size: int = 32, lr: float = 1e-3) -> Dict[str, Any]:
    set_seed(seed)
    model = make_cnn(n_classes, factors["init"], factors["batch_norm"], factors["dropout"], size)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=factors["weight_decay"])
    (Xtr, ytr), (Xva, yva), (Xte, yte) = data["train"], data["val"], data["test"]
    g = torch.Generator().manual_seed(seed)
    hist = {"train_loss": [], "val_acc": []}
    for _ in range(epochs):
        model.train()
        order, run = torch.randperm(len(Xtr), generator=g), 0.0
        for s in range(0, len(Xtr), batch_size):
            idx = order[s:s + batch_size]
            xb = augment_batch(Xtr[idx], g) if factors["augmentation"] else Xtr[idx]
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), ytr[idx])
            loss.backward()
            opt.step()
            run += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            hist["train_loss"].append(run / len(Xtr))
            hist["val_acc"].append(float((model(Xva).argmax(1) == yva).float().mean()))
    with torch.no_grad():
        pred = model(Xte).argmax(1).numpy()
    rep = classification_report(yte.numpy(), pred, [str(i) for i in range(n_classes)])
    return {"factors": factors, "final_train_loss": hist["train_loss"][-1], "best_val_acc": max(hist["val_acc"]),
            "test_acc": rep["accuracy"], "test_macro_f1": rep["macro_f1"], "history": hist}


def run_ablations(cfg: Optional[Dict[str, Any]] = None, synthetic: bool = False, epochs: Optional[int] = None,
                  size: int = 32, per_class: Optional[int] = None) -> Dict[str, Any]:
    cfg = cfg or load_config()
    ec, seed = cfg["experiments"], cfg["project"]["seed"]
    epochs = epochs or ec["epochs"]
    data = load_splits(cfg, size, synthetic, per_class)
    n = len(cfg["classes"]["names"])
    grid = {"init": ec["inits"], "batch_norm": ec["batch_norm"], "weight_decay": ec["weight_decay"],
            "dropout": ec["dropout"], "augmentation": ec["augmentation"]}
    results: Dict[str, List[Dict[str, Any]]] = {}
    for factor, values in grid.items():
        results[factor] = []
        for v in values:
            r = run_one({**BASELINE, factor: v}, data, n, epochs, seed, size)
            r["value"] = v
            results[factor].append(r)
            logger.info("%-13s=%-6s test_acc=%.3f val_acc=%.3f", factor, v, r["test_acc"], r["best_val_acc"])
    fig, ax = plt.subplots(2, 5, figsize=(20, 7))
    for j, (factor, rs) in enumerate(results.items()):
        labels = [str(r["value"]) for r in rs]
        ax[0, j].bar(labels, [r["test_acc"] for r in rs], color="tab:blue")
        ax[0, j].set(title=f"{factor}: test accuracy", ylim=(0, 1))
        for r, lab in zip(rs, labels):
            ax[1, j].plot(r["history"]["train_loss"], label=lab)
        ax[1, j].set(title=f"{factor}: train loss", xlabel="epoch")
        ax[1, j].legend(fontsize=8)
    fig_path = save_figure(fig, "ablation_summary", "experiments")
    summary = {"baseline": BASELINE, "epochs": epochs, "figure": str(fig_path),
               "results": {f: [{"value": r["value"], **{k: r[k] for k in ("test_acc", "test_macro_f1", "best_val_acc", "final_train_loss")}}
                               for r in rs] for f, rs in results.items()},
               "histories": {f: {str(r["value"]): r["history"] for r in rs} for f, rs in results.items()}}
    save_json(summary, "ablations", "experiments")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Controlled ablation experiments")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--size", type=int, default=32)
    ap.add_argument("--per-class", type=int, default=None)
    a = ap.parse_args()
    s = run_ablations(None, a.synthetic, a.epochs, a.size, a.per_class)
    print(f"saved {s['figure']}")


if __name__ == "__main__":
    main()
