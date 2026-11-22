"""Unit II (c): numerical (finite-difference) vs analytical gradient checking with error plots."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np

from src.utils import get_logger, save_figure, save_json

logger = get_logger(__name__)


def numerical_gradient(f: Callable[[np.ndarray], float], x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Central-difference gradient ``(f(x+h) - f(x-h)) / 2h``, one coordinate at a time (``O(h^2)`` error)."""
    x = x.astype(np.float64)
    g = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        old = x[idx]
        x[idx] = old + eps
        fp = f(x)
        x[idx] = old - eps
        fm = f(x)
        x[idx] = old
        g[idx] = (fp - fm) / (2 * eps)
    return g


def relative_error(a: np.ndarray, b: np.ndarray, floor: float = 1e-12) -> float:
    """``||a-b|| / max(||a|| + ||b||, floor)`` (scale-free, robust to tiny gradients)."""
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a) + np.linalg.norm(b), floor))


@dataclass
class GradCheckResult:
    """Result of a gradient check."""

    rel_error: float
    max_abs_diff: float
    passed: bool
    tolerance: float


def check_gradient(f: Callable[[np.ndarray], float], grad_f: Callable[[np.ndarray], np.ndarray], x: np.ndarray,
                   eps: float = 1e-5, tol: float = 1e-6) -> GradCheckResult:
    """Compare ``grad_f(x)`` with the numerical gradient of ``f`` at ``x``."""
    ga, gn = grad_f(x.astype(np.float64)), numerical_gradient(f, x, eps)
    rel = relative_error(ga, gn)
    return GradCheckResult(rel, float(np.max(np.abs(ga - gn))), rel < tol, tol)


def check_param_dict_gradients(loss_fn: Callable[[], float], params: Dict[str, np.ndarray],
                               grads: Mapping[str, np.ndarray], eps: float = 1e-5, tol: float = 1e-6
                               ) -> Dict[str, GradCheckResult]:
    """Check analytical grads for every array in a parameter dict (e.g. an MLP's ``W1, b1, ...``).

    ``loss_fn`` must read the *current* values in ``params`` (arrays are perturbed in place).
    """
    out: Dict[str, GradCheckResult] = {}
    for name, p in params.items():
        gn = np.zeros_like(p)
        it = np.nditer(p, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            old = p[idx]
            p[idx] = old + eps
            fp = loss_fn()
            p[idx] = old - eps
            fm = loss_fn()
            p[idx] = old
            gn[idx] = (fp - fm) / (2 * eps)
        rel = relative_error(grads[name], gn)
        out[name] = GradCheckResult(rel, float(np.max(np.abs(grads[name] - gn))), rel < tol, tol)
    return out


def plot_gradient_check(f: Callable[[np.ndarray], float], grad_f: Callable[[np.ndarray], np.ndarray], x: np.ndarray,
                        name: str = "gradient_check", eps_list: Optional[np.ndarray] = None) -> str:
    """Save analytical-vs-numerical scatter and the error-vs-step-size U-curve (truncation vs round-off)."""
    eps_list = np.logspace(-12, -1, 12) if eps_list is None else eps_list
    ga = grad_f(x)
    errs = [relative_error(ga, numerical_gradient(f, x, e)) for e in eps_list]
    gn = numerical_gradient(f, x, 1e-5)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].scatter(ga.ravel(), gn.ravel(), s=14)
    lim = [min(ga.min(), gn.min()), max(ga.max(), gn.max())]
    ax[0].plot(lim, lim, "r--"); ax[0].set(title="Analytical vs numerical (eps=1e-5)", xlabel="analytical", ylabel="numerical")
    ax[1].loglog(eps_list, errs, "o-"); ax[1].axhline(1e-6, c="g", ls="--", label="tol 1e-6")
    ax[1].set(title="Relative error vs step size", xlabel="eps", ylabel="relative error"); ax[1].legend()
    return str(save_figure(fig, name, subdir="math"))


def demo() -> Dict[str, object]:
    """Check softmax cross-entropy on a linear model in float64 and plot the diagnostics."""
    rng = np.random.default_rng(0)
    X, W_true = rng.normal(size=(40, 6)), rng.normal(size=(6, 4))
    y = np.argmax(X @ W_true, axis=1)

    def loss(w: np.ndarray) -> float:
        z = X @ w.reshape(6, 4)
        z -= z.max(1, keepdims=True)
        lp = z - np.log(np.exp(z).sum(1, keepdims=True))
        return float(-lp[np.arange(len(y)), y].mean())

    def grad(w: np.ndarray) -> np.ndarray:
        z = X @ w.reshape(6, 4)
        z -= z.max(1, keepdims=True)
        p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
        p[np.arange(len(y)), y] -= 1
        return (X.T @ p / len(y)).ravel()

    w0 = rng.normal(size=24) * 0.3
    res = check_gradient(loss, grad, w0)
    fig = plot_gradient_check(loss, grad, w0, "gradient_check_softmax")
    out = {"rel_error": res.rel_error, "passed": res.passed, "figure": fig}
    save_json(out, "gradient_check_demo")
    return out


if __name__ == "__main__":
    print(demo())
