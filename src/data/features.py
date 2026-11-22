"""Hand-crafted image features for the from-scratch baseline MLP (colour + gradient statistics)."""
from __future__ import annotations

import numpy as np


def extract_features(img: np.ndarray, bins: int = 8) -> np.ndarray:
    """Convert an ``(H, W, 3)`` uint8 image into a ``3*2 + 3*bins + 2`` feature vector.

    Features: per-channel mean/std, per-channel intensity histograms (normalised), mean gradient
    magnitude and the foreground fraction (pixels far from the belt grey background).
    """
    x = img.astype(np.float32)
    mean, std = x.mean(axis=(0, 1)) / 255.0, x.std(axis=(0, 1)) / 255.0
    hists = [np.histogram(x[..., c], bins=bins, range=(0, 255))[0] / x[..., c].size for c in range(3)]
    gray = x.mean(axis=2)
    gy, gx = np.gradient(gray)
    grad = float(np.hypot(gx, gy).mean() / 255.0)
    fg = float((np.abs(x - 70.0).max(axis=2) > 35).mean())
    return np.concatenate([mean, std, *hists, [grad, fg]]).astype(np.float64)


def batch_features(images: np.ndarray, bins: int = 8) -> np.ndarray:
    """Vectorise :func:`extract_features` over ``(N, H, W, 3)``."""
    return np.stack([extract_features(im, bins) for im in images])
