"""Unit I (d): compare Batch GD, SGD, Momentum and NAG on the same baseline waste-classification task.

Baseline task: 6-class waste-material classification from hand-crafted colour/gradient features of
real TrashNet images, using identical MLP initialisation for every optimizer.
Also draws optimizer trajectories on an ill-conditioned 2-D quadratic.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.data.features import batch_features
from src.data.loaders import load_trashnet_arrays
from src.fundamentals.mlp import MLP, TrainingLogger, fit
from src.fundamentals.optimizers import make_optimizer
from src.utils import get_logger, load_config, resolve_path, save_figure, save_json, set_seed

logger = get_logger(__name__)


def make_baseline_task(cfg: Optional[Dict[str, Any]] = None, per_class: Optional[int] = None, size: int = 32,
                       allow_synthetic: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Standardised ``(X_train, y_train, X_val, y_val)`` features from REAL TrashNet images.

    Uses the leakage-safe train/val folders if ``python -m src.data.pipeline --source trashnet`` was run,
    else a seeded random split of the raw folder. Statistics come from the training set only.
    ``allow_synthetic=True`` is for unit tests only.
    """
    cfg = cfg or load_config()
    seed = cfg["project"]["seed"]
    names = cfg["classes"]["names"]
    proc = resolve_path(cfg["paths"]["processed_root"]) / "trashnet" / "classification"
    if (proc / "train").is_dir() and (proc / "val").is_dir():
        itr, ytr = load_trashnet_arrays(cfg, size, "train", per_class, allow_synthetic, seed)
        iva, yva = load_trashnet_arrays(cfg, size, "val", per_class, allow_synthetic, seed)
        Xtr, Xva = batch_features(itr), batch_features(iva)
    else:
        imgs, y = load_trashnet_arrays(cfg, size, None, per_class, allow_synthetic, seed)
        X = batch_features(imgs)
        n_val = int(len(X) * cfg["fundamentals"]["training"]["val_fraction"])
        Xva, yva, Xtr, ytr = X[:n_val], y[:n_val], X[n_val:], y[n_val:]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    return (Xtr - mu) / sd, ytr, (Xva - mu) / sd, yva


def run_comparison(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Train one identically-initialised MLP per optimizer; save curves + JSON summary."""
    cfg = cfg or load_config()
    seed = cfg["project"]["seed"]
    set_seed(seed)
    fm, tr = cfg["fundamentals"]["mlp"], cfg["fundamentals"]["training"]
    Xtr, ytr, Xva, yva = make_baseline_task(cfg)
    n_in, n_out = Xtr.shape[1], len(cfg["classes"]["names"])
    runs: Dict[str, TrainingLogger] = {}
    for key, hp in cfg["fundamentals"]["optimizers"].items():
        model = MLP([n_in, *fm["hidden_sizes"], n_out], fm["hidden_activation"], fm["init"], fm["loss"], fm["l2"], seed)
        opt = make_optimizer(key, **hp)
        runs[opt.name] = fit(model, Xtr, ytr, opt, tr["epochs"], tr["batch_size"], Xva, yva, seed)
        logger.info("%-9s final train_loss=%.4f val_acc=%.3f", opt.name, runs[opt.name].history[-1]["train_loss"],
                    runs[opt.name].history[-1]["val_acc"])
        runs[opt.name].save(f"optimizer_{key}")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for name, lg in runs.items():
        ax[0].plot(lg.column("epoch"), lg.column("train_loss"), label=name)
        ax[1].plot(lg.column("updates"), lg.column("train_loss"), label=name)
        ax[2].plot(lg.column("epoch"), lg.column("val_acc"), label=name)
    ax[0].set(title="Train loss vs epoch", xlabel="epoch", ylabel="cross-entropy", yscale="log")
    ax[1].set(title="Train loss vs #parameter updates", xlabel="updates", ylabel="cross-entropy", yscale="log", xscale="log")
    ax[2].set(title="Validation accuracy", xlabel="epoch", ylabel="accuracy")
    ax[0].legend()
    p = save_figure(fig, "optimizer_comparison", subdir="fundamentals")
    summary = {"figure": str(p), "trajectory_figure": toy_trajectories(),
               "results": {n: {k: lg.history[-1][k] for k in ("train_loss", "train_acc", "val_loss", "val_acc", "time_s", "updates")}
                           for n, lg in runs.items()}}
    save_json(summary, "optimizer_comparison_summary", "fundamentals")
    return summary


def toy_trajectories(steps: int = 40) -> str:
    """Optimizer paths on f(x, y) = 0.5 (x^2 + 20 y^2) (ill-conditioned valley); saves a contour plot."""
    A = np.array([1.0, 20.0])
    f = lambda p: 0.5 * float(np.sum(A * p ** 2))  # noqa: E731
    cfgs = {"Batch GD": ("batch_gd", {"lr": 0.09}), "SGD": ("sgd", {"lr": 0.09}),
            "Momentum": ("momentum", {"lr": 0.02, "beta": 0.9}), "NAG": ("nag", {"lr": 0.02, "beta": 0.9})}
    gx, gy = np.meshgrid(np.linspace(-3.5, 3.5, 200), np.linspace(-1.5, 1.5, 200))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.contour(gx, gy, 0.5 * (A[0] * gx ** 2 + A[1] * gy ** 2), levels=20, cmap="Greys", alpha=0.6)
    for label, (key, hp) in cfgs.items():
        opt = make_optimizer(key, **hp)
        params = {"p": np.array([-3.0, 1.2])}
        path = [params["p"].copy()]
        for _ in range(steps):
            opt.step(params, {"p": A * params["p"]})
            path.append(params["p"].copy())
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], "o-", ms=2, lw=1.2, label=f"{label} (f={f(path[-1]):.1e})")
    ax.set(title=f"Optimizer trajectories after {steps} steps", xlabel="x", ylabel="y"); ax.legend(fontsize=8)
    return str(save_figure(fig, "optimizer_trajectories", subdir="fundamentals"))


if __name__ == "__main__":
    import json

    print(json.dumps(run_comparison(), indent=2))
