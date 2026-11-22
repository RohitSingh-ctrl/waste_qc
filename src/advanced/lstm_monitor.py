"""Unit V (a): LSTM defect-trend monitor for the conveyor stream.

Idea: every time window (e.g. one minute of belt) the sorter outputs how many items of each class it saw. From the last
``seq_len`` windows an LSTM predicts the *next* window's contamination rate (share of non-recyclable items in the
recyclable stream) and raises an alert when the prediction crosses ``spike_threshold``.

IMPORTANT (honesty note): there is no public time-series dataset for this task, so the stream is *simulated*. The class
mix comes from the real processed training split (when available) and contamination episodes are injected on top
(gradual ramp, plateau, decay). Everything saved by this module is tagged ``"simulated_stream": true``. The numbers
show that the monitoring pipeline works and how it compares with simple baselines; they are NOT field performance.
Replace :func:`simulate_stream` output with real per-window class counts to evaluate on real logs.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from src.utils import get_logger, load_config, resolve_path, save_figure, save_json, set_seed

logger = get_logger(__name__)


# ----------------------------------------------------------------------------- stream simulation
def class_priors(cfg: Dict[str, Any], synthetic: bool = False) -> np.ndarray:
    """Class mix of the stream.

    Real mode: image counts per class in the processed *train* split (raises ``DataNotFoundError`` if missing).
    Synthetic mode: uniform, with rare classes down-weighted by ``data.synthetic.rare_class_weight``.
    """
    names = cfg["classes"]["names"]
    if synthetic:
        w = np.ones(len(names))
        for i, n in enumerate(names):
            if n in cfg["classes"]["rare"]:
                w[i] = cfg["data"]["synthetic"]["rare_class_weight"]
        return w / w.sum()
    from src.vision.common import classification_root
    root = classification_root(cfg, False) / "train"
    counts = np.array([len([p for p in (root / n).glob("*.*")]) if (root / n).is_dir() else 0 for n in names], float)
    if counts.sum() == 0:
        from src.data.loaders import DataNotFoundError
        raise DataNotFoundError(f"No training images under {root}")
    return counts / counts.sum()


def simulate_stream(priors: Sequence[float], contaminant_ids: Sequence[int], n_steps: int = 2000,
                    items_per_step: int = 40, episode_every: int = 150, seed: int = 42) -> Dict[str, np.ndarray]:
    """Simulate per-window class fractions with injected contamination episodes.

    Returns:
        ``fractions`` ``[n_steps, n_classes]``, ``rate`` ``[n_steps]`` (contaminant share), ``episode`` ``[n_steps]`` bool.
    """
    rng = np.random.default_rng(seed)
    pri = np.asarray(priors, float)
    pri = pri / pri.sum()
    cont = np.zeros(len(pri), bool)
    cont[list(contaminant_ids)] = True
    base_cont = float(pri[cont].sum())
    boost = np.zeros(n_steps)
    n_ep = max(1, n_steps // episode_every)
    for k in range(n_ep):
        start = int(k * episode_every + rng.integers(10, max(11, episode_every // 2)))
        ramp, hold, decay = int(rng.integers(4, 9)), int(rng.integers(8, 25)), int(rng.integers(4, 9))
        peak = float(rng.uniform(0.35, 0.70))
        shape = np.concatenate([np.linspace(0, peak, ramp), np.full(hold, peak), np.linspace(peak, 0, decay)])
        end = min(n_steps, start + len(shape))
        if end > start:
            boost[start:end] = np.maximum(boost[start:end], shape[: end - start])
    fractions = np.zeros((n_steps, len(pri)))
    for t in range(n_steps):
        p_cont = base_cont + boost[t] * (1.0 - base_cont)
        p = np.where(cont, pri / max(pri[cont].sum(), 1e-12) * p_cont, pri / max(pri[~cont].sum(), 1e-12) * (1 - p_cont))
        fractions[t] = rng.multinomial(items_per_step, p / p.sum()) / items_per_step
    return {"fractions": fractions, "rate": fractions[:, cont].sum(1), "episode": boost > 0}


def make_windows(fractions: np.ndarray, rate: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """``X[i] = fractions[i : i+L]``, ``y[i] = rate[i+L]`` (the next window's contamination rate)."""
    n = len(rate) - seq_len
    X = np.stack([fractions[i:i + seq_len] for i in range(n)])
    return X.astype(np.float32), rate[seq_len:seq_len + n].astype(np.float32)


def chronological_split(X: np.ndarray, y: np.ndarray, seq_len: int, ratios=(0.7, 0.15, 0.15)):
    """Time-ordered split with a purge gap of ``seq_len`` windows so no input window overlaps a later split's targets."""
    n = len(X)
    a, b = int(n * ratios[0]), int(n * (ratios[0] + ratios[1]))
    return {"train": (X[:a], y[:a]), "val": (X[a + seq_len:b], y[a + seq_len:b]), "test": (X[b + seq_len:], y[b + seq_len:])}


# ----------------------------------------------------------------------------- model
class LSTMTrend(nn.Module):
    """Stacked LSTM -> last hidden state -> sigmoid (predicted contamination rate in [0, 1])."""

    def __init__(self, n_features: int, hidden: int = 64, layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return torch.sigmoid(self.head(out[:, -1])).squeeze(-1)


def train_lstm(data: Dict[str, Tuple[np.ndarray, np.ndarray]], hidden: int, layers: int, epochs: int, lr: float = 1e-2,
               batch_size: int = 64, seed: int = 42) -> Tuple[LSTMTrend, Dict[str, List[float]]]:
    """Train with MSE, keep the weights of the best validation epoch."""
    set_seed(seed)
    Xtr, ytr = map(torch.from_numpy, data["train"])
    Xva, yva = map(torch.from_numpy, data["val"])
    model = LSTMTrend(Xtr.shape[-1], hidden, layers)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    hist: Dict[str, List[float]] = {"train_mse": [], "val_mse": []}
    best, best_state = float("inf"), None
    for ep in range(1, epochs + 1):
        model.train()
        order, run = torch.randperm(len(Xtr), generator=g), 0.0
        for s in range(0, len(Xtr), batch_size):
            idx = order[s:s + batch_size]
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(Xtr[idx]), ytr[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            val = float(nn.functional.mse_loss(model(Xva), yva))
        hist["train_mse"].append(run / len(Xtr))
        hist["val_mse"].append(val)
        if val < best:
            best, best_state = val, {k: v.clone() for k, v in model.state_dict().items()}
        logger.info("epoch %d/%d train_mse=%.5f val_mse=%.5f", ep, epochs, hist["train_mse"][-1], val)
    model.load_state_dict(best_state)
    return model.eval(), hist


# ----------------------------------------------------------------------------- evaluation
def spike_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> Dict[str, float]:
    """Alert quality: a *spike* is a next-window rate above ``threshold``."""
    t, p = y_true > threshold, y_pred > threshold
    tp, fp, fn = int((t & p).sum()), int((~t & p).sum()), int((t & ~p).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "true_spikes": int(t.sum()), "alerts": int(p.sum()), "tp": tp, "fp": fp, "fn": fn}


def evaluate_forecasters(model: LSTMTrend, X: np.ndarray, y: np.ndarray, cont_ids: Sequence[int], threshold: float
                         ) -> Dict[str, Any]:
    """Compare the LSTM with two baselines: persistence (repeat last rate) and a 5-window moving average."""
    rate_in = X[:, :, list(cont_ids)].sum(-1)
    with torch.no_grad():
        lstm = model(torch.from_numpy(X)).numpy()
    preds = {"lstm": lstm, "persistence": rate_in[:, -1], "moving_avg_5": rate_in[:, -5:].mean(1)}
    out = {k: {"mae": float(np.abs(v - y).mean()), **spike_metrics(y, v, threshold)} for k, v in preds.items()}
    return {"metrics": out, "predictions": preds}


def run_lstm_monitor(cfg: Optional[Dict[str, Any]] = None, synthetic: bool = False, epochs: Optional[int] = None,
                     n_steps: int = 2000) -> Dict[str, Any]:
    """Simulate stream -> windows -> chronological split -> train LSTM -> compare with baselines -> save figure/JSON/ckpt."""
    cfg = cfg or load_config()
    ac = cfg["advanced"]["lstm"]
    names, seed = cfg["classes"]["names"], cfg["project"]["seed"]
    cont_ids = [names.index(c) for c in cfg["classes"]["contaminant"]]
    priors = class_priors(cfg, synthetic)
    stream = simulate_stream(priors, cont_ids, n_steps=n_steps, seed=seed)
    X, y = make_windows(stream["fractions"], stream["rate"], ac["seq_len"])
    data = chronological_split(X, y, ac["seq_len"])
    epochs = epochs or ac["epochs"]
    model, hist = train_lstm(data, ac["hidden"], ac["layers"], epochs, seed=seed)
    Xte, yte = data["test"]
    ev = evaluate_forecasters(model, Xte, yte, cont_ids, ac["spike_threshold"])
    fig, ax = plt.subplots(1, 2, figsize=(13, 3.8))
    ax[0].plot(hist["train_mse"], label="train"); ax[0].plot(hist["val_mse"], label="val")
    ax[0].set(title="LSTM loss (MSE)", xlabel="epoch"); ax[0].legend()
    ax[1].plot(yte, color="k", lw=1, label="true next-window rate")
    ax[1].plot(ev["predictions"]["lstm"], lw=1, label="LSTM")
    ax[1].plot(ev["predictions"]["persistence"], lw=1, ls=":", label="persistence")
    ax[1].axhline(ac["spike_threshold"], color="r", ls="--", label="alert threshold")
    ax[1].set(title="Contamination trend, test segment (SIMULATED stream)", xlabel="window", ylabel="contamination rate")
    ax[1].legend(fontsize=8)
    fig_path = save_figure(fig, "lstm_trend_monitor", "advanced", cfg=cfg)
    ckpt_dir = resolve_path(cfg["paths"]["checkpoints"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "n_features": X.shape[-1], "hidden": ac["hidden"], "layers": ac["layers"],
                "seq_len": ac["seq_len"], "classes": names}, ckpt_dir / "lstm_monitor.pt")
    result = {"simulated_stream": True, "class_priors": dict(zip(names, map(float, priors))), "n_steps": n_steps,
              "seq_len": ac["seq_len"], "epochs": epochs, "spike_threshold": ac["spike_threshold"],
              "n_windows": {k: len(v[0]) for k, v in data.items()}, "history": hist, "test": ev["metrics"],
              "figure": str(fig_path)}
    save_json(result, "lstm_monitor", "advanced", cfg=cfg)
    for k, m in ev["metrics"].items():
        logger.info("%-13s MAE=%.4f spike P/R/F1=%.2f/%.2f/%.2f (true spikes=%d)", k, m["mae"], m["precision"], m["recall"],
                    m["f1"], m["true_spikes"])
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="LSTM defect-trend monitor (simulated conveyor stream)")
    ap.add_argument("--synthetic", action="store_true", help="uniform class mix instead of the real train split")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    run_lstm_monitor(load_config(a.config), a.synthetic, a.epochs, a.steps)


if __name__ == "__main__":
    main()
