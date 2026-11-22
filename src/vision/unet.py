"""Unit IV (b): U-Net semantic segmentation (written from scratch with nn.Module) for waste regions."""
from __future__ import annotations

from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.evaluation.benchmark import count_parameters, measure_latency, model_size_mb
from src.evaluation.metrics import segmentation_metrics
from src.utils import get_logger, load_config, save_figure, save_json, set_seed
from src.vision.common import Timer, checkpoint_path, detection_tag, get_device, parse_common_args, processed_dir
from src.vision.datasets import SemanticSegmentation

logger = get_logger(__name__)


def double_conv(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class UNet(nn.Module):
    """Encoder (maxpool + double-conv) / decoder (transposed conv + skip concatenation).

    ``n_classes`` includes background (class 0). Input H/W must be divisible by ``2**depth``.
    """

    def __init__(self, n_classes: int, base: int = 16, depth: int = 4, in_ch: int = 3) -> None:
        super().__init__()
        chs = [base * 2 ** i for i in range(depth + 1)]
        self.inc = double_conv(in_ch, chs[0])
        self.down = nn.ModuleList([nn.Sequential(nn.MaxPool2d(2), double_conv(chs[i], chs[i + 1])) for i in range(depth)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(chs[i + 1], chs[i], 2, stride=2) for i in reversed(range(depth))])
        self.dec = nn.ModuleList([double_conv(chs[i] * 2, chs[i]) for i in reversed(range(depth))])
        self.head = nn.Conv2d(chs[0], n_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = [self.inc(x)]
        for d in self.down:
            skips.append(d(skips[-1]))
        x = skips.pop()
        for up, dec in zip(self.up, self.dec):
            x = dec(torch.cat([up(x), skips.pop()], dim=1))
        return self.head(x)


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Soft multi-class Dice loss (mean over classes)."""
    n = logits.shape[1]
    p = F.softmax(logits, 1)
    t = F.one_hot(target, n).permute(0, 3, 1, 2).float()
    inter, denom = (p * t).sum((0, 2, 3)), p.sum((0, 2, 3)) + t.sum((0, 2, 3))
    return 1.0 - ((2 * inter + eps) / (denom + eps)).mean()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, n_classes: int, device: torch.device) -> Dict[str, Any]:
    model.eval()
    ps, ts = [], []
    for x, y in loader:
        ps.append(model(x.to(device)).argmax(1).cpu().numpy())
        ts.append(y.numpy())
    return segmentation_metrics(np.concatenate(ps), np.concatenate(ts), n_classes)


def train_unet(cfg: Optional[Dict[str, Any]] = None, synthetic: bool = False, epochs: Optional[int] = None,
               image_size: int = 128) -> Dict[str, Any]:
    cfg = cfg or load_config()
    vc = cfg["vision"]["segmentation"]
    set_seed(cfg["project"]["seed"])
    device, n_classes = get_device(), len(cfg["classes"]["names"]) + 1
    root = processed_dir(cfg, detection_tag(cfg, synthetic), "semantic")
    dl = {s: DataLoader(SemanticSegmentation(root, s, image_size, augment=True), vc["batch_size"], shuffle=(s == "train"))
          for s in ("train", "val", "test")}
    model = UNet(n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=vc["lr"])
    epochs = epochs or vc["epochs"]
    hist = {"train_loss": [], "val_miou": []}
    with Timer() as t:
        for ep in range(1, epochs + 1):
            model.train()
            run, n = 0.0, 0
            for x, y in dl["train"]:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                out = model(x)
                loss = F.cross_entropy(out, y) + dice_loss(out, y)
                loss.backward()
                opt.step()
                run, n = run + loss.item() * len(y), n + len(y)
            m = evaluate(model, dl["val"], n_classes, device)
            hist["train_loss"].append(run / n)
            hist["val_miou"].append(m["miou"])
            logger.info("epoch %d/%d loss=%.4f val_mIoU=%.3f", ep, epochs, hist["train_loss"][-1], m["miou"])
    test = evaluate(model, dl["test"], n_classes, device)
    sample = next(iter(dl["test"]))[0][:1].to(device)
    model.eval()
    with torch.no_grad():
        lat = measure_latency(lambda: model(sample), warmup=2, runs=10)
    result = {"model": "unet", "epochs": epochs, "train_seconds": t.seconds, "params": count_parameters(model),
              "size_mb": model_size_mb(model), **lat, "history": hist, "test": test}
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    ax[0].plot(hist["train_loss"]); ax[0].set(title="U-Net train loss (CE + Dice)", xlabel="epoch")
    ax[1].plot(hist["val_miou"]); ax[1].set(title="U-Net val mIoU", xlabel="epoch")
    save_figure(fig, "unet_curves", "vision")
    _plot_predictions(model, dl["test"], device)
    torch.save({"model": "unet", "state_dict": model.state_dict(), "n_classes": n_classes}, checkpoint_path(cfg, "unet"))
    save_json(result, "segmentation_unet", "vision")
    logger.info("U-Net test mIoU=%.3f dice=%.3f", test["miou"], test["mean_dice"])
    return result


@torch.no_grad()
def _plot_predictions(model: nn.Module, loader: DataLoader, device: torch.device, k: int = 4) -> None:
    model.eval()
    x, y = next(iter(loader))
    pred = model(x.to(device)).argmax(1).cpu()
    k = min(k, len(x))
    mean, std = torch.tensor([0.485, 0.456, 0.406])[:, None, None], torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    fig, ax = plt.subplots(3, k, figsize=(3 * k, 8), squeeze=False)
    for i in range(k):
        ax[0, i].imshow((x[i] * std + mean).clamp(0, 1).permute(1, 2, 0)); ax[0, i].set_title("image")
        ax[1, i].imshow(y[i], vmin=0, vmax=6, cmap="tab10"); ax[1, i].set_title("ground truth")
        ax[2, i].imshow(pred[i], vmin=0, vmax=6, cmap="tab10"); ax[2, i].set_title("U-Net")
        [a.axis("off") for a in ax[:, i]]
    save_figure(fig, "unet_predictions", "vision")


def main() -> None:
    a = parse_common_args("Train U-Net").parse_args()
    train_unet(load_config(a.config), a.synthetic, a.epochs, a.image_size or 128)


if __name__ == "__main__":
    main()
