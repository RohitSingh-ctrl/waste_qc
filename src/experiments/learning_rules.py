"""Unit III: classical learning rules (NumPy only): Hebbian, Oja, Competitive, Error-Correction and an RBM (CD-1)."""
from __future__ import annotations

from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np

from src.fundamentals.activations import sigmoid
from src.utils import save_figure, save_json


def _correlated_data(n: int, rng: np.random.Generator) -> np.ndarray:
    cov = np.array([[3.0, 1.6], [1.6, 1.0]])
    X = rng.multivariate_normal([0, 0], cov, n)
    return X - X.mean(0)


def hebbian_demo(seed: int = 0, eta: float = 0.01, steps: int = 400) -> Dict[str, Any]:
    """Plain Hebb ``dw = eta*y*x`` (y = w.x): weights align with the principal axis but the norm diverges."""
    rng = np.random.default_rng(seed)
    X = _correlated_data(steps, rng)
    w, norms = rng.normal(0, 0.1, 2), []
    for x in X:
        w = w + eta * (w @ x) * x
        norms.append(float(np.linalg.norm(w)))
    return {"w": w, "norms": np.array(norms), "X": X}


def oja_demo(seed: int = 0, eta: float = 0.01, epochs: int = 20) -> Dict[str, Any]:
    """Oja ``dw = eta*y*(x - y*w)``: norm self-stabilises to 1 and w -> first principal component."""
    rng = np.random.default_rng(seed)
    X = _correlated_data(500, rng)
    w, norms = rng.normal(0, 0.1, 2), []
    for _ in range(epochs):
        for x in X[rng.permutation(len(X))]:
            y = w @ x
            w = w + eta * y * (x - y * w)
            norms.append(float(np.linalg.norm(w)))
    evals, evecs = np.linalg.eigh(np.cov(X.T))
    pc1 = evecs[:, -1]
    return {"w": w, "norms": np.array(norms), "pc1": pc1, "cosine": float(abs(w @ pc1) / np.linalg.norm(w)), "X": X}


def competitive_demo(seed: int = 0, k: int = 3, eta: float = 0.05, epochs: int = 15) -> Dict[str, Any]:
    """Winner-take-all: only the nearest prototype moves toward the sample."""
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0, 0.0], [4.0, 4.0], [-4.0, 4.0]])[:k]
    X = np.vstack([c + rng.normal(0, 0.6, (100, 2)) for c in centers])
    W = X[rng.choice(len(X), k, replace=False)].copy()
    for _ in range(epochs):
        for x in X[rng.permutation(len(X))]:
            win = np.argmin(((W - x) ** 2).sum(1))
            W[win] += eta * (x - W[win])
    err = float(np.mean(np.min(np.linalg.norm(W[:, None] - centers[None], axis=-1), axis=0)))
    return {"W": W, "X": X, "centers": centers, "center_error": err}


def error_correction_demo(seed: int = 0, eta: float = 0.1, epochs: int = 15) -> Dict[str, Any]:
    """Perceptron/error-correction rule ``w += eta*(t - y)*x`` on linearly separable data: errors fall to 0."""
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal([2, 2], 0.8, (60, 2)), rng.normal([-2, -2], 0.8, (60, 2))])
    t = np.r_[np.ones(60), np.zeros(60)]
    w, b, errors = np.zeros(2), 0.0, []
    for _ in range(epochs):
        e = 0
        for i in rng.permutation(len(X)):
            y = float(w @ X[i] + b >= 0)
            if y != t[i]:
                w, b, e = w + eta * (t[i] - y) * X[i], b + eta * (t[i] - y), e + 1
        errors.append(e)
    return {"w": w, "b": b, "errors": errors, "X": X, "t": t}


def bars_and_stripes(n: int = 4) -> np.ndarray:
    """All ``n x n`` binary images that are pure rows (stripes) or pure columns (bars)."""
    pats = set()
    for mask in range(2 ** n):
        v = np.array([(mask >> i) & 1 for i in range(n)])
        pats.add(tuple(np.tile(v[:, None], (1, n)).ravel()))
        pats.add(tuple(np.tile(v[None, :], (n, 1)).ravel()))
    return np.array(sorted(pats), dtype=float)


def rbm_demo(seed: int = 0, n_hidden: int = 16, epochs: int = 2000, eta: float = 0.1) -> Dict[str, Any]:
    """Bernoulli RBM trained with contrastive divergence (CD-1) on 4x4 bars-and-stripes."""
    rng = np.random.default_rng(seed)
    V = bars_and_stripes(4)
    nv = V.shape[1]
    W, bv, bh = rng.normal(0, 0.1, (nv, n_hidden)), np.zeros(nv), np.zeros(n_hidden)
    errs = []
    for _ in range(epochs):
        ph = sigmoid(V @ W + bh)
        h = (rng.random(ph.shape) < ph).astype(float)
        pv = sigmoid(h @ W.T + bv)
        v1 = (rng.random(pv.shape) < pv).astype(float)
        ph1 = sigmoid(v1 @ W + bh)
        W += eta * (V.T @ ph - v1.T @ ph1) / len(V)
        bv += eta * (V - v1).mean(0)
        bh += eta * (ph - ph1).mean(0)
        errs.append(float(np.mean((V - sigmoid(sigmoid(V @ W + bh) @ W.T + bv)) ** 2)))
    return {"W": W, "errors": errs, "V": V, "recon": sigmoid(sigmoid(V @ W + bh) @ W.T + bv)}


def run_all() -> Dict[str, Any]:
    heb, oja, comp, ec, rbm = hebbian_demo(), oja_demo(), competitive_demo(), error_correction_demo(), rbm_demo()
    fig, ax = plt.subplots(2, 4, figsize=(19, 8))
    ax[0, 0].plot(heb["norms"]); ax[0, 0].set(title="Hebbian: |w| diverges", xlabel="step", yscale="log")
    ax[0, 1].scatter(*heb["X"].T, s=6, alpha=0.4); ax[0, 1].quiver(0, 0, *heb["w"] / np.linalg.norm(heb["w"]) * 3,
                                                                    color="r", angles="xy", scale_units="xy", scale=1)
    ax[0, 1].set(title="Hebbian: w aligns with max variance", aspect="equal")
    ax[0, 2].plot(oja["norms"]); ax[0, 2].axhline(1, color="k", ls="--"); ax[0, 2].set(title=f"Oja: |w|->1 (cos to PC1={oja['cosine']:.3f})")
    ax[0, 3].scatter(*oja["X"].T, s=6, alpha=0.4)
    for v, c, lab in ((oja["pc1"], "g", "PCA PC1"), (oja["w"], "r", "Oja w")):
        ax[0, 3].quiver(0, 0, *v * 3, color=c, angles="xy", scale_units="xy", scale=1, label=lab)
    ax[0, 3].legend(); ax[0, 3].set(title="Oja vs PCA", aspect="equal")
    ax[1, 0].scatter(*comp["X"].T, s=6, alpha=0.4); ax[1, 0].scatter(*comp["W"].T, c="r", marker="x", s=120)
    ax[1, 0].set(title=f"Competitive: prototypes (err={comp['center_error']:.2f})")
    ax[1, 1].plot(ec["errors"], marker="o"); ax[1, 1].set(title="Error-correction: mistakes per epoch", xlabel="epoch")
    ax[1, 2].plot(rbm["errors"]); ax[1, 2].set(title="RBM (CD-1): reconstruction MSE", xlabel="epoch")
    tiles = rbm["W"].T.reshape(-1, 4, 4)[:8]
    ax[1, 3].imshow(np.vstack([np.hstack(list(tiles[:4])), np.hstack(list(tiles[4:8]))]), cmap="RdBu"); ax[1, 3].set(title="RBM hidden-unit filters")
    ax[1, 3].axis("off")
    path = save_figure(fig, "learning_rules", "experiments")
    summary = {"hebbian_final_norm": float(heb["norms"][-1]), "oja_final_norm": float(oja["norms"][-1]),
               "oja_cosine_to_pc1": oja["cosine"], "competitive_center_error": comp["center_error"],
               "error_correction_errors": ec["errors"], "rbm_final_mse": rbm["errors"][-1], "figure": str(path)}
    save_json(summary, "learning_rules", "experiments")
    return summary


if __name__ == "__main__":
    print(run_all())
