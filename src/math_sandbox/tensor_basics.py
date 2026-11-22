"""Unit II (a): tensor creation, broadcasting, inner products, reshaping/striding on waste images.

Only raw ``torch.Tensor`` operations are used (no ``torch.nn``, no autograd).
"""
from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.loaders import load_trashnet_arrays
from src.utils import get_logger, save_figure, set_seed

logger = get_logger(__name__)


# ---------------------------------------------------------------- creation
def tensor_creation_demo(seed: int = 42) -> Dict[str, torch.Tensor]:
    """Create tensors with every common constructor; returns them keyed by name."""
    g = torch.Generator().manual_seed(seed)
    return {
        "zeros": torch.zeros(2, 3),
        "ones": torch.ones(2, 3),
        "arange": torch.arange(0, 12, dtype=torch.float32),
        "linspace": torch.linspace(0, 1, 5),
        "eye": torch.eye(3),
        "randn": torch.randn(2, 3, generator=g),
        "from_numpy": torch.from_numpy(np.arange(6, dtype=np.float64).reshape(2, 3)),
        "image_batch": torch.rand(4, 3, 8, 8, generator=g),  # (N, C, H, W)
    }


# ---------------------------------------------------------------- broadcasting
def broadcast_shape(a: Sequence[int], b: Sequence[int]) -> Tuple[int, ...]:
    """Compute the broadcast shape by hand using the NumPy/PyTorch rules.

    Rule: align shapes on the right; dimensions are compatible if equal or one of them is 1.

    Raises:
        ValueError: if shapes are not broadcast-compatible.
    """
    out = []
    for i in range(1, max(len(a), len(b)) + 1):
        da = a[-i] if i <= len(a) else 1
        db = b[-i] if i <= len(b) else 1
        if da != db and da != 1 and db != 1:
            raise ValueError(f"Shapes {tuple(a)} and {tuple(b)} are not broadcastable (dim -{i}: {da} vs {db})")
        out.append(max(da, db))
    return tuple(reversed(out))


def broadcasting_demo() -> Dict[str, Any]:
    """Verify the manual rule against torch and normalise an image batch per channel by broadcasting."""
    x = torch.arange(6.0).reshape(2, 3)
    cases = [((2, 3), (3,)), ((4, 1, 5), (3, 1)), ((8, 3, 16, 16), (3, 1, 1)), ((2, 3), (2,))]
    results = []
    for sa, sb in cases:
        try:
            manual = broadcast_shape(sa, sb)
            torch_shape = tuple((torch.zeros(sa) + torch.zeros(sb)).shape)
            results.append({"a": sa, "b": sb, "manual": manual, "torch": torch_shape, "match": manual == torch_shape})
        except ValueError:
            try:
                torch.zeros(sa) + torch.zeros(sb)
                match = False
            except RuntimeError:
                match = True  # both reject the pair
            results.append({"a": sa, "b": sb, "manual": "error", "torch": "error", "match": match})
    return {"cases": results, "row_plus_col": (x[:, :1] + x[:1, :]).tolist()}


def channel_normalisation_figure(seed: int = 42, allow_synthetic: bool = False) -> str:
    """Per-channel (mean/std) normalisation of waste images via broadcasting; saves a before/after figure."""
    set_seed(seed)
    imgs, labels = load_trashnet_arrays(None, 64, None, 2, allow_synthetic, seed)
    x = torch.from_numpy(imgs[:6]).permute(0, 3, 1, 2).float() / 255.0          # (N, C, H, W)
    mean = x.mean(dim=(0, 2, 3), keepdim=True)                                  # (1, C, 1, 1)
    std = x.std(dim=(0, 2, 3), keepdim=True)                                    # (1, C, 1, 1)
    z = (x - mean) / std                                                        # broadcast (N,C,H,W) - (1,C,1,1)
    fig, axes = plt.subplots(2, 6, figsize=(12, 4.2))
    for i in range(6):
        axes[0, i].imshow(x[i].permute(1, 2, 0).numpy())
        axes[1, i].imshow(((z[i] - z[i].min()) / (z[i].max() - z[i].min())).permute(1, 2, 0).numpy())
        axes[0, i].axis("off"); axes[1, i].axis("off")
    axes[0, 0].set_title("raw", loc="left"); axes[1, 0].set_title("(x-mean)/std via broadcasting", loc="left")
    fig.suptitle(f"Broadcasting: {tuple(x.shape)} - {tuple(mean.shape)}")
    return str(save_figure(fig, "math_broadcast_normalisation", subdir="math"))


# ---------------------------------------------------------------- inner products
def dot_product_demo(seed: int = 42) -> Dict[str, float]:
    """Dot product computed 4 ways (loop, sum of products, torch.dot, einsum) plus cosine similarity."""
    g = torch.Generator().manual_seed(seed)
    a, b = torch.randn(256, generator=g, dtype=torch.float64), torch.randn(256, generator=g, dtype=torch.float64)
    loop = 0.0
    for i in range(a.numel()):
        loop += float(a[i] * b[i])
    res = {"loop": loop, "sum_prod": float((a * b).sum()), "torch_dot": float(torch.dot(a, b)),
           "einsum": float(torch.einsum("i,i->", a, b)),
           "cosine": float(torch.dot(a, b) / (a.norm() * b.norm()))}
    return res


def matmul_as_inner_products(seed: int = 42) -> float:
    """Show ``(A @ B)[i, j] == <A[i, :], B[:, j]>``; returns the max abs deviation."""
    g = torch.Generator().manual_seed(seed)
    A, B = torch.randn(4, 5, generator=g, dtype=torch.float64), torch.randn(5, 3, generator=g, dtype=torch.float64)
    manual = torch.stack([torch.stack([torch.dot(A[i], B[:, j]) for j in range(B.shape[1])]) for i in range(A.shape[0])])
    return float((manual - A @ B).abs().max())


# ---------------------------------------------------------------- reshaping / striding
def striding_demo() -> Dict[str, Any]:
    """Views vs copies, strides, transpose contiguity and ``as_strided`` sliding windows."""
    x = torch.arange(12.0).reshape(3, 4)
    t = x.t()
    v = x.view(4, 3)
    v[0, 0] = 100.0  # shares storage with x
    shares = bool(x[0, 0] == 100.0)
    x[0, 0] = 0.0
    windows = torch.as_strided(torch.arange(8.0), size=(6, 3), stride=(1, 1))  # overlapping windows of length 3
    return {"stride_x": x.stride(), "stride_t": t.stride(), "t_contiguous": t.is_contiguous(),
            "view_shares_storage": shares, "reshape_of_transpose_copies": t.reshape(12).data_ptr() != x.data_ptr(),
            "sliding_windows": windows.tolist()}


def run_all(allow_synthetic: bool = False) -> Dict[str, Any]:
    """Run every tensor demo, save the figure, and return the results."""
    out = {"creation": {k: tuple(v.shape) for k, v in tensor_creation_demo().items()},
           "broadcasting": broadcasting_demo(), "dot": dot_product_demo(),
           "matmul_deviation": matmul_as_inner_products(), "striding": striding_demo(),
           "figure": channel_normalisation_figure(allow_synthetic=allow_synthetic)}
    logger.info("Tensor demos done: %s", out["figure"])
    return out


if __name__ == "__main__":
    import pprint

    pprint.pprint(run_all())
