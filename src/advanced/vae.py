"""Unit V (b): Convolutional VAE.

Three experiments (all on small images, CPU-friendly):

* ``compare``  - same VAE, hidden bottleneck activation = step / sigmoid / tanh (identical data, seed, epochs).
                 The step function has zero gradient almost everywhere, so it is trained with a *straight-through
                 estimator* (forward: hard step, backward: identity clipped to ``|x| <= 1``). This is a known workaround,
                 not a free lunch; the comparison shows what it costs.
* ``latent2d`` - VAE with a 2-D latent space; test images are encoded and plotted coloured by class.
* ``generate`` - VAE trained on one rare class (default ``trash``); samples are saved as a grid. Intended as a
                 candidate for augmenting under-represented classes; with ~100 images, expect blurry samples.

Real TrashNet is required unless ``--synthetic`` is given (smoke tests only).
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
from src.utils import get_logger, load_config, resolve_path, save_figure, save_json, set_seed
from src.vision.common import get_device

logger = get_logger(__name__)


# ----------------------------------------------------------------------------- activations
class StepSTE(torch.autograd.Function):
    """Hard step (x > 0) forward; clipped-identity gradient backward (straight-through estimator)."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        ctx.save_for_backward(x)
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, g: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        (x,) = ctx.saved_tensors
        return g * (x.abs() <= 1).to(g.dtype)


class Step(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return StepSTE.apply(x)


def make_activation(name: str) -> nn.Module:
    acts = {"step": Step, "sigmoid": nn.Sigmoid, "tanh": nn.Tanh, "relu": nn.ReLU}
    if name not in acts:
        raise ValueError(f"unknown activation '{name}' (choose from {list(acts)})")
    return acts[name]()


# ----------------------------------------------------------------------------- model
class ConvVAE(nn.Module):
    """Encoder: 3 stride-2 convs -> FC hidden (selectable activation) -> (mu, logvar). Decoder mirrors it.

    ``size`` must be divisible by 8.
    """

    def __init__(self, latent_dim: int = 16, size: int = 32, hidden_activation: str = "relu", base: int = 16,
                 fc_hidden: int = 128) -> None:
        super().__init__()
        if size % 8:
            raise ValueError("size must be divisible by 8")
        self.size, self.latent_dim, self.s8 = size, latent_dim, size // 8
        c = [3, base, base * 2, base * 4]
        self.enc = nn.Sequential(*[m for i in range(3) for m in (nn.Conv2d(c[i], c[i + 1], 4, 2, 1), nn.ReLU())], nn.Flatten())
        flat = c[3] * self.s8 ** 2
        self.fc_h = nn.Linear(flat, fc_hidden)
        self.act = make_activation(hidden_activation)
        self.fc_mu, self.fc_lv = nn.Linear(fc_hidden, latent_dim), nn.Linear(fc_hidden, latent_dim)
        self.fc_d = nn.Linear(latent_dim, flat)
        self.dec = nn.Sequential(*[m for i in (3, 2) for m in (nn.ConvTranspose2d(c[i], c[i - 1], 4, 2, 1), nn.ReLU())],
                                 nn.ConvTranspose2d(c[1], 3, 4, 2, 1), nn.Sigmoid())
        self.c3 = c[3]

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.act(self.fc_h(self.enc(x)))
        return self.fc_mu(h), self.fc_lv(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec(F.relu(self.fc_d(z)).view(-1, self.c3, self.s8, self.s8))

    def forward(self, x: torch.Tensor):
        mu, lv = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv) if self.training else mu
        return self.decode(z), mu, lv


def vae_loss(recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float = 1.0
             ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-sample ELBO terms: summed-pixel MSE reconstruction + beta * KL(q(z|x) || N(0, I))."""
    rec = F.mse_loss(recon, x, reduction="sum") / len(x)
    kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1)).mean()
    return rec + beta * kl, rec, kl


# ----------------------------------------------------------------------------- data
def load_images(cfg: Dict[str, Any], size: int, split: str, per_class: Optional[int], synthetic: bool,
                only_class: Optional[str] = None) -> Tuple[torch.Tensor, np.ndarray]:
    """Images in ``[0, 1]`` as ``[N, 3, size, size]`` plus integer labels (config class order)."""
    names, seed = cfg["classes"]["names"], cfg["project"]["seed"]
    if synthetic:   # different seed per split -> no train/test overlap even in smoke tests
        imgs, y = make_classification_arrays(per_class or 20, size, names, seed + {"train": 0, "val": 100, "test": 200}[split])
    else:
        imgs, y = load_trashnet_arrays(cfg, size, split, per_class, False, seed)
    if only_class is not None:
        keep = y == names.index(only_class)
        imgs, y = imgs[keep], y[keep]
    return torch.from_numpy(imgs).permute(0, 3, 1, 2).float() / 255.0, np.asarray(y)


# ----------------------------------------------------------------------------- training
def train_vae(Xtr: torch.Tensor, Xva: torch.Tensor, latent_dim: int, size: int, activation: str, epochs: int,
              beta: float = 1.0, batch_size: int = 32, lr: float = 1e-3, seed: int = 42) -> Tuple[ConvVAE, Dict[str, List[float]]]:
    set_seed(seed)
    device = get_device()
    model = ConvVAE(latent_dim, size, activation).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    hist: Dict[str, List[float]] = {"train_loss": [], "val_recon": [], "val_kl": []}
    Xva_d = Xva.to(device)
    for ep in range(1, epochs + 1):
        model.train()
        order, run = torch.randperm(len(Xtr), generator=g), 0.0
        for s in range(0, len(Xtr), batch_size):
            xb = Xtr[order[s:s + batch_size]].to(device)
            opt.zero_grad()
            recon, mu, lv = model(xb)
            loss, _, _ = vae_loss(recon, xb, mu, lv, beta)
            loss.backward()
            opt.step()
            run += loss.item() * len(xb)
        model.eval()
        with torch.no_grad():
            recon, mu, lv = model(Xva_d)
            _, rec, kl = vae_loss(recon, Xva_d, mu, lv, beta)
        hist["train_loss"].append(run / len(Xtr))
        hist["val_recon"].append(float(rec) / Xva[0].numel())     # per-pixel MSE
        hist["val_kl"].append(float(kl))
        logger.info("[%s z=%d] epoch %d/%d loss=%.3f val_pixel_mse=%.5f val_kl=%.3f", activation, latent_dim, ep, epochs,
                    hist["train_loss"][-1], hist["val_recon"][-1], hist["val_kl"][-1])
    return model.eval(), hist


def _grid(imgs: torch.Tensor, ncol: int) -> np.ndarray:
    imgs = imgs.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
    rows = [np.concatenate(list(imgs[i:i + ncol]), 1) for i in range(0, len(imgs) - ncol + 1, ncol)]
    return np.concatenate(rows, 0)


# ----------------------------------------------------------------------------- experiments
def run_activation_comparison(cfg: Optional[Dict[str, Any]] = None, synthetic: bool = False, epochs: Optional[int] = None,
                              size: int = 32, per_class: Optional[int] = None,
                              activations: Tuple[str, ...] = ("step", "sigmoid", "tanh")) -> Dict[str, Any]:
    """Train one VAE per bottleneck activation under identical conditions and compare validation reconstruction."""
    cfg = cfg or load_config()
    vc, seed = cfg["advanced"]["vae"], cfg["project"]["seed"]
    epochs = epochs or vc["epochs"]
    Xtr, _ = load_images(cfg, size, "train", per_class, synthetic)
    Xva, _ = load_images(cfg, size, "val", per_class, synthetic)
    results, models = {}, {}
    for a in activations:
        m, h = train_vae(Xtr, Xva, vc["latent_dim"], size, a, epochs, vc["beta"], vc["batch_size"], seed=seed)
        models[a], results[a] = m, {"history": h, "final_val_pixel_mse": h["val_recon"][-1], "final_val_kl": h["val_kl"][-1]}
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    for a in activations:
        ax[0].plot(results[a]["history"]["train_loss"], label=a)
        ax[1].plot(results[a]["history"]["val_recon"], label=a)
        ax[2].plot(results[a]["history"]["val_kl"], label=a)
    for k, t in enumerate(("train loss (ELBO)", "val pixel MSE", "val KL")):
        ax[k].set(title=t, xlabel="epoch"); ax[k].legend()
    save_figure(fig, "vae_activation_curves", "advanced", cfg=cfg)
    device = get_device()
    sample = Xva[:8].to(device)
    fig2, ax2 = plt.subplots(len(activations) + 1, 1, figsize=(10, 1.6 * (len(activations) + 1)))
    ax2[0].imshow(_grid(Xva[:8], 8)); ax2[0].set_title("input", fontsize=9)
    for i, a in enumerate(activations, 1):
        with torch.no_grad():
            ax2[i].imshow(_grid(models[a](sample)[0], 8)); ax2[i].set_title(f"reconstruction - {a}", fontsize=9)
    for x in ax2:
        x.axis("off")
    fig_path = save_figure(fig2, "vae_activation_reconstructions", "advanced", cfg=cfg)
    out = {"epochs": epochs, "size": size, "latent_dim": vc["latent_dim"], "synthetic": synthetic,
           "note": "step uses a straight-through estimator", "results": results, "figure": str(fig_path)}
    save_json(out, "vae_activation_comparison", "advanced", cfg=cfg)
    return out


def run_latent_2d(cfg: Optional[Dict[str, Any]] = None, synthetic: bool = False, epochs: Optional[int] = None,
                  size: int = 32, per_class: Optional[int] = None) -> Dict[str, Any]:
    """Train a VAE with ``latent_dim=2`` and plot the test-set posterior means coloured by class."""
    cfg = cfg or load_config()
    vc, seed, names = cfg["advanced"]["vae"], cfg["project"]["seed"], cfg["classes"]["names"]
    epochs = epochs or vc["epochs"]
    Xtr, _ = load_images(cfg, size, "train", per_class, synthetic)
    Xva, _ = load_images(cfg, size, "val", per_class, synthetic)
    Xte, yte = load_images(cfg, size, "test", per_class, synthetic)
    model, hist = train_vae(Xtr, Xva, 2, size, "tanh", epochs, vc["beta"], vc["batch_size"], seed=seed)
    device = get_device()
    with torch.no_grad():
        mu = model.encode(Xte.to(device))[0].cpu().numpy()
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for i, n in enumerate(names):
        pts = mu[yte == i]
        ax[0].scatter(pts[:, 0], pts[:, 1], s=10, label=n, alpha=0.7)
    ax[0].set(title="2-D latent space (test posterior means)", xlabel="z1", ylabel="z2"); ax[0].legend(fontsize=8)
    lin = np.linspace(-2.5, 2.5, 10)
    zs = torch.tensor([[a, b] for b in lin[::-1] for a in lin], dtype=torch.float32, device=device)
    with torch.no_grad():
        ax[1].imshow(_grid(model.decode(zs), 10))
    ax[1].set(title="decoded latent grid (z in [-2.5, 2.5]^2)"); ax[1].axis("off")
    fig_path = save_figure(fig, "vae_latent_2d", "advanced", cfg=cfg)
    sep = {n: mu[yte == i].mean(0).tolist() for i, n in enumerate(names) if (yte == i).any()}
    out = {"epochs": epochs, "synthetic": synthetic, "class_centroids": sep, "history": hist, "figure": str(fig_path)}
    save_json(out, "vae_latent_2d", "advanced", cfg=cfg)
    return out


def run_generate_rare(cfg: Optional[Dict[str, Any]] = None, synthetic: bool = False, epochs: Optional[int] = None,
                      size: int = 32, per_class: Optional[int] = None, class_name: Optional[str] = None,
                      n_samples: int = 16) -> Dict[str, Any]:
    """Train on one rare class and sample new images (augmentation candidate)."""
    cfg = cfg or load_config()
    vc, seed = cfg["advanced"]["vae"], cfg["project"]["seed"]
    class_name = class_name or cfg["classes"]["rare"][0]
    epochs = epochs or vc["epochs"]
    Xtr, _ = load_images(cfg, size, "train", per_class, synthetic, class_name)
    Xva, _ = load_images(cfg, size, "val", per_class, synthetic, class_name)
    if len(Xtr) < 2 or len(Xva) < 1:
        raise ValueError(f"not enough '{class_name}' images (train={len(Xtr)}, val={len(Xva)})")
    model, hist = train_vae(Xtr, Xva, vc["latent_dim"], size, "tanh", epochs, vc["beta"], min(vc["batch_size"], len(Xtr)), seed=seed)
    device = get_device()
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        gen = model.decode(torch.randn(n_samples, vc["latent_dim"], generator=g).to(device))
    ncol = 8 if n_samples >= 8 else n_samples
    fig, ax = plt.subplots(2, 1, figsize=(9, 3.2))
    ax[0].imshow(_grid(Xtr[:ncol], ncol)); ax[0].set_title(f"real '{class_name}' (train)", fontsize=9)
    ax[1].imshow(_grid(gen[:ncol * (n_samples // ncol)], ncol)); ax[1].set_title("VAE samples", fontsize=9)
    for x in ax:
        x.axis("off")
    fig_path = save_figure(fig, f"vae_generated_{class_name}", "advanced", cfg=cfg)
    out = {"class": class_name, "n_train_images": len(Xtr), "epochs": epochs, "synthetic": synthetic,
           "final_val_pixel_mse": hist["val_recon"][-1], "figure": str(fig_path)}
    save_json(out, f"vae_generate_{class_name}", "advanced", cfg=cfg)
    ckpt_dir = resolve_path(cfg["paths"]["checkpoints"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "size": size, "latent_dim": vc["latent_dim"], "class": class_name},
               ckpt_dir / f"vae_{class_name}.pt")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Convolutional VAE experiments")
    ap.add_argument("--task", choices=["compare", "latent2d", "generate", "all"], default="all")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--image-size", type=int, default=32)
    ap.add_argument("--per-class", type=int, default=None, help="limit images per class (faster CPU runs)")
    ap.add_argument("--class-name", default=None, help="class for --task generate (default: first rare class)")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    kw = dict(synthetic=a.synthetic, epochs=a.epochs, size=a.image_size, per_class=a.per_class)
    if a.task in ("compare", "all"):
        run_activation_comparison(cfg, **kw)
    if a.task in ("latent2d", "all"):
        run_latent_2d(cfg, **kw)
    if a.task in ("generate", "all"):
        run_generate_rare(cfg, class_name=a.class_name, **kw)


if __name__ == "__main__":
    main()
