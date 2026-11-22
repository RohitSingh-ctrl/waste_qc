"""Unit II (b): SVD for waste-image feature compression and rank diagnostics.

Provides: SVD reconstruction, Eckart-Young error check, energy/rank selection, a from-scratch
power-iteration SVD (for pedagogy) and the diagnostic plots.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.data.loaders import load_trashnet_arrays
from src.utils import get_logger, save_figure, save_json, set_seed

logger = get_logger(__name__)


def svd_decompose(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Thin SVD ``X = U diag(S) Vt``."""
    return np.linalg.svd(X, full_matrices=False)


def reconstruct(U: np.ndarray, S: np.ndarray, Vt: np.ndarray, rank: int) -> np.ndarray:
    """Rank-``k`` reconstruction ``U_k S_k V_k^T``."""
    return (U[:, :rank] * S[:rank]) @ Vt[:rank]


def reconstruction_errors(X: np.ndarray, ranks: Sequence[int]) -> Dict[str, List[float]]:
    """Frobenius reconstruction error per rank, relative error, and the Eckart-Young theoretical value."""
    U, S, Vt = svd_decompose(X)
    fro = float(np.linalg.norm(X))
    emp, theory = [], []
    for k in ranks:
        emp.append(float(np.linalg.norm(X - reconstruct(U, S, Vt, k))))
        theory.append(float(np.sqrt(np.sum(S[k:] ** 2))))  # Eckart-Young: error^2 = sum of discarded sigma^2
    return {"ranks": list(ranks), "error": emp, "theory": theory, "relative": [e / fro for e in emp]}


def rank_for_energy(S: np.ndarray, energy: float = 0.95) -> int:
    """Smallest rank whose cumulative squared singular values reach ``energy`` of the total."""
    cum = np.cumsum(S ** 2) / np.sum(S ** 2)
    return int(np.searchsorted(cum, energy) + 1)


def power_iteration_svd(X: np.ndarray, k: int, iters: int = 300, seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """From-scratch top-``k`` SVD by power iteration with deflation (pedagogical, O(k * iters * nm))."""
    rng = np.random.default_rng(seed)
    A = X.astype(np.float64).copy()
    U, S, V = [], [], []
    for _ in range(k):
        v = rng.normal(size=A.shape[1])
        v /= np.linalg.norm(v)
        for _ in range(iters):
            v = A.T @ (A @ v)
            n = np.linalg.norm(v)
            if n == 0:
                break
            v /= n
        sigma = np.linalg.norm(A @ v)
        u = (A @ v) / sigma
        U.append(u); S.append(sigma); V.append(v)
        A -= sigma * np.outer(u, v)  # deflate
    return np.stack(U, 1), np.asarray(S), np.stack(V, 0)


def build_feature_matrix(n_per_class: int = 30, size: int = 32, seed: int = 42,
                         allow_synthetic: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Flattened grayscale REAL TrashNet images ``(N, size*size)`` (mean-centred) and labels."""
    imgs, y = load_trashnet_arrays(None, size, None, n_per_class, allow_synthetic, seed)
    X = imgs.astype(np.float64).mean(axis=3).reshape(len(imgs), -1) / 255.0
    return X - X.mean(axis=0, keepdims=True), y


def run_svd_study(n_per_class: int = 30, size: int = 32, seed: int = 42, energy: float = 0.95,
                  allow_synthetic: bool = False) -> Dict[str, object]:
    """Full study: spectrum, cumulative energy, error-vs-rank, visual reconstructions, power-iteration check."""
    set_seed(seed)
    X, y = build_feature_matrix(n_per_class, size, seed, allow_synthetic)
    U, S, Vt = svd_decompose(X)
    ranks = [1, 2, 5, 10, 20, 40, min(X.shape)]
    errs = reconstruction_errors(X, ranks)
    k95 = rank_for_energy(S, energy)

    # figure 1: spectrum + energy + error
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    ax[0].semilogy(S, "o-", ms=3); ax[0].set(title="Singular value spectrum", xlabel="index", ylabel="sigma")
    cum = np.cumsum(S ** 2) / np.sum(S ** 2)
    ax[1].plot(cum); ax[1].axhline(energy, ls="--", c="r"); ax[1].axvline(k95 - 1, ls="--", c="r")
    ax[1].set(title=f"Cumulative energy (rank {k95} -> {energy:.0%})", xlabel="rank", ylabel="energy")
    ax[2].plot(errs["ranks"], errs["error"], "o-", label="empirical")
    ax[2].plot(errs["ranks"], errs["theory"], "x--", label="Eckart-Young")
    ax[2].set(title="Reconstruction error vs rank", xlabel="rank", ylabel="Frobenius error"); ax[2].legend()
    p1 = save_figure(fig, "svd_rank_diagnostics", subdir="math")

    # figure 2: reconstructions
    show = [1, 5, 20, k95]
    fig, axes = plt.subplots(3, len(show) + 1, figsize=(2.2 * (len(show) + 1), 6.6))
    for r, i in enumerate([0, 1, 2]):
        axes[r, 0].imshow(X[i].reshape(size, size), cmap="gray"); axes[r, 0].set_title("original" if r == 0 else "")
        for c, k in enumerate(show, 1):
            axes[r, c].imshow(reconstruct(U, S, Vt, k)[i].reshape(size, size), cmap="gray")
            if r == 0:
                axes[r, c].set_title(f"rank {k}")
    for a in axes.ravel():
        a.axis("off")
    p2 = save_figure(fig, "svd_reconstructions", subdir="math")

    # figure 3: 2-D projection onto top singular directions coloured by class
    Z = U[:, :2] * S[:2]
    fig, ax = plt.subplots(figsize=(5, 4.2))
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=y, cmap="tab10", s=12)
    ax.set(title="Top-2 SVD projection (colour = class)", xlabel="PC1", ylabel="PC2"); fig.colorbar(sc)
    p3 = save_figure(fig, "svd_projection", subdir="math")

    # from-scratch check
    _, S_pi, _ = power_iteration_svd(X, 5)
    pi_err = float(np.max(np.abs(S_pi - S[:5]) / S[:5]))
    summary = {"matrix_shape": X.shape, "rank_for_95pct_energy": k95, "numerical_rank": int(np.sum(S > S[0] * 1e-10)),
               "compression_ratio_at_k95": float(X.size / (k95 * (sum(X.shape) + 1))),
               "eckart_young_max_gap": float(np.max(np.abs(np.array(errs["error"]) - np.array(errs["theory"])))),
               "power_iteration_rel_err": pi_err, "figures": [str(p1), str(p2), str(p3)]}
    save_json(summary, "svd_summary")
    logger.info("SVD: %s", {k: v for k, v in summary.items() if k != "figures"})
    return summary


if __name__ == "__main__":
    print(run_svd_study())
