"""Unit IV: 2-D and 3-D convolution written by hand (NumPy loops/vectorised windows) and visualised.

The manual implementations are verified against ``torch.nn.functional.conv2d/conv3d`` (cross-correlation convention).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

from src.data.synthetic import render_single_item
from src.utils import load_config, resolve_path, save_figure, save_json

KERNELS = {
    "identity": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], float),
    "box blur": np.ones((3, 3)) / 9.0,
    "gaussian": np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], float) / 16.0,
    "sobel-x": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float),
    "sobel-y": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], float),
    "laplacian": np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], float),
    "sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], float),
}


def conv2d(img: np.ndarray, kernel: np.ndarray, stride: int = 1, padding: int = 0) -> np.ndarray:
    """Cross-correlation. ``img`` (H,W) or (C,H,W) with ``kernel`` (kh,kw) or (C,kh,kw) -> output (H',W')."""
    img = img[None] if img.ndim == 2 else img
    kernel = kernel[None] if kernel.ndim == 2 else kernel
    img = np.pad(img, ((0, 0), (padding, padding), (padding, padding)))
    C, H, W = img.shape
    kh, kw = kernel.shape[1:]
    oh, ow = (H - kh) // stride + 1, (W - kw) // stride + 1
    out = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            out[i, j] = np.sum(img[:, i * stride:i * stride + kh, j * stride:j * stride + kw] * kernel)
    return out


def conv3d(vol: np.ndarray, kernel: np.ndarray, padding: int = 0) -> np.ndarray:
    """3-D cross-correlation over (D,H,W) with a (kd,kh,kw) kernel (stride 1)."""
    vol = np.pad(vol, padding)
    kd, kh, kw = kernel.shape
    D, H, W = vol.shape
    out = np.zeros((D - kd + 1, H - kh + 1, W - kw + 1))
    for d in range(out.shape[0]):
        for i in range(out.shape[1]):
            for j in range(out.shape[2]):
                out[d, i, j] = np.sum(vol[d:d + kd, i:i + kh, j:j + kw] * kernel)
    return out


def verify_against_torch(seed: int = 0) -> Dict[str, float]:
    import torch
    import torch.nn.functional as F
    rng = np.random.default_rng(seed)
    x, k = rng.normal(size=(3, 12, 12)), rng.normal(size=(3, 3, 3))
    a = conv2d(x, k, stride=2, padding=1)
    b = F.conv2d(torch.tensor(x)[None], torch.tensor(k)[None], stride=2, padding=1)[0, 0].numpy()
    v, k3 = rng.normal(size=(6, 10, 10)), rng.normal(size=(3, 3, 3))
    c = conv3d(v, k3, padding=1)
    d = F.conv3d(torch.tensor(v)[None, None], torch.tensor(k3)[None, None], padding=1)[0, 0].numpy()
    return {"conv2d_max_abs_diff": float(np.abs(a - b).max()), "conv3d_max_abs_diff": float(np.abs(c - d).max())}


def _sample_image(cfg: Dict[str, Any], size: int, synthetic: bool) -> np.ndarray:
    """First real TrashNet/TACO-crop image if present, otherwise a synthetic item (only with ``synthetic=True``)."""
    from PIL import Image
    if not synthetic:
        for sub in ("trashnet/classification", "taco/crops"):
            root = resolve_path(cfg["paths"]["processed_root"]) / sub
            files = sorted(root.glob("*/*/*.*")) if root.is_dir() else []
            if files:
                return np.asarray(Image.open(files[0]).convert("RGB").resize((size, size)), dtype=float) / 255.0
        raise FileNotFoundError("No processed images found. Run the data pipeline, or pass --synthetic.")
    return render_single_item("plastic", size, np.random.default_rng(0)).astype(float) / 255.0


def run_visualizations(cfg: Optional[Dict[str, Any]] = None, synthetic: bool = False, size: int = 64) -> Dict[str, Any]:
    cfg = cfg or load_config()
    img = _sample_image(cfg, size, synthetic)
    gray = img @ np.array([0.299, 0.587, 0.114])
    # --- 2D: classic kernels on grayscale
    fig, ax = plt.subplots(2, 4, figsize=(14, 7))
    ax[0, 0].imshow(img); ax[0, 0].set_title("input (RGB)")
    for a, (name, k) in zip(ax.ravel()[1:], KERNELS.items()):
        a.imshow(conv2d(gray, k, padding=1), cmap="gray"); a.set_title(name)
    [a.axis("off") for a in ax.ravel()]
    p2 = save_figure(fig, "conv2d_filters", "experiments")
    # --- 2D multi-channel: one 3x3x3 kernel sums over RGB -> one feature map
    rgb_edge = np.stack([KERNELS["sobel-x"]] * 3) / 3.0
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.8))
    ax[0].imshow(img); ax[0].set_title("RGB input (3 channels)")
    ax[1].imshow(rgb_edge.reshape(9, 3), cmap="coolwarm"); ax[1].set_title("3x3x3 kernel (unrolled)")
    ax[2].imshow(conv2d(img.transpose(2, 0, 1), rgb_edge, padding=1), cmap="gray"); ax[2].set_title("feature map (sum over C)")
    [a.axis("off") for a in ax]
    p2b = save_figure(fig, "conv2d_multichannel", "experiments")
    # --- 3D: volume = frames that shift/darken over depth; depth-difference kernel responds to *change* over depth
    frames = np.stack([np.roll(gray, 3 * d, axis=1) * (1.0 - 0.08 * d) for d in range(6)])
    k_time = np.zeros((3, 3, 3)); k_time[0, 1, 1], k_time[2, 1, 1] = -1.0, 1.0
    k_blur = np.ones((3, 3, 3)) / 27.0
    out_t, out_b = conv3d(frames, k_time, padding=1), conv3d(frames, k_blur, padding=1)
    fig, ax = plt.subplots(3, 6, figsize=(16, 8))
    for d in range(6):
        ax[0, d].imshow(frames[d], cmap="gray"); ax[0, d].set_title(f"depth {d}")
        ax[1, d].imshow(out_t[d], cmap="coolwarm"); ax[2, d].imshow(out_b[d], cmap="gray")
    ax[1, 0].set_ylabel("depth-diff kernel"); ax[2, 0].set_ylabel("3D box blur")
    [a.set_xticks([]) or a.set_yticks([]) for a in ax.ravel()]
    p3 = save_figure(fig, "conv3d_volume", "experiments")
    out = {**verify_against_torch(), "figures": [str(p2), str(p2b), str(p3)]}
    save_json(out, "conv_visualization", "experiments")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    print(run_visualizations(None, ap.parse_args().synthetic))
