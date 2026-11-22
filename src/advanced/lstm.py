"""Unit V (a): LSTM defect-trend monitor.

Forecasts the next batch's defect (contamination) rate from the last ``seq_len`` batches and raises an alert when the
forecast reaches ``spike_threshold``. Splits are chronological (no shuffling), the scaler uses training data only, and
the model is compared against a persistence baseline (``next = last``) and a moving average.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from src.advanced.defect_stream import load_defect_csv, simulate_defect_series
from src.utils import ensure_dir, get_logger, load_config, save_figure, save_json, set_seed
from src.vision.common import Timer, checkpoint_path, get_device

logger = get_logger(__name__)


class DefectLSTM(nn.Module):
    """Stacked LSTM -> small MLP head; input ``(B, L, 1)`` normalised rates, output ``(B,)``."""

    def __init__(self, hidden: int = 64, layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)


# ------------------------------------------------------------------------------------------- data
def make_windows(z: np.ndarray, seq_len: int, horizon: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sliding windows. Returns ``X (N, seq_len, 1)``, ``y (N,)`` and the target index of every window.

    The window for target ``t`` only contains values up to ``t - horizon`` (no future information).
    """
    idx = np.arange(seq_len + horizon - 1, len(z))
    X = np.stack([z[t - horizon + 1 - seq_len: t - horizon + 1] for t in idx])[..., None]
    return X.astype(np.float32), z[idx].astype(np.float32), idx


def chrono_split(n: int, ratios=(0.7, 0.15, 0.15)) -> Tuple[int, int]:
    """Return the end index of the train and validation segments."""
    tr = int(n * ratios[0])
    return tr, tr + int(n * ratios[1])


def _alert_metrics(y_true: np.ndarray, y_pred: np.ndarray, thr: float) -> Dict[str, Any]:
    spike, alert = y_true >= thr, y_pred >= thr
    tp, fp, fn = int((spike & alert).sum()), int((~spike & alert).sum()), int((spike & ~alert).sum())
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    f1 = 2 * prec * rec / (prec + rec) if prec and rec else (0.0 if prec is not None and rec is not None else None)
    # event level: a spike run counts as caught if an alert fires within 3 batches before it or during it
    events, caught, leads = 0, 0, []
    i = 0
    while i < len(spike):
        if spike[i]:
            j = i
            while j + 1 < len(spike) and spike[j + 1]:
                j += 1
            events += 1
            lo = max(0, i - 3)
            hits = np.where(alert[lo:j + 1])[0]
            if len(hits):
                caught += 1
                leads.append(int(i - (lo + hits[0])))
            i = j + 1
        else:
            i += 1
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "spike_events": events,
            "events_caught": caught, "mean_lead_batches": float(np.mean(leads)) if leads else None}


def _regression(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    return {"mae": float(np.mean(np.abs(y - p))), "rmse": float(np.sqrt(np.mean((y - p) ** 2)))}


# ------------------------------------------------------------------------------------------- inference helpers
def load_lstm(path: Path | str, device: str = "cpu") -> Tuple[DefectLSTM, Dict[str, Any]]:
    """Load a checkpoint saved by :func:`train_lstm` -> ``(model, meta)``."""
    meta = torch.load(path, map_location=device, weights_only=False)
    m = DefectLSTM(meta["hidden"], meta["layers"], meta["dropout"]).to(device)
    m.load_state_dict(meta["state_dict"])
    return m.eval(), meta


@torch.no_grad()
def rolling_forecast(model: DefectLSTM, meta: Dict[str, Any], rates: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Forecast every batch ``t >= seq_len + horizon - 1``. Returns an array aligned with ``rates`` (NaN before)."""
    rates = np.asarray(rates, dtype=np.float32)
    z = (rates - meta["mean"]) / meta["std"]
    out = np.full(len(rates), np.nan, dtype=np.float32)
    if len(rates) < meta["seq_len"] + meta["horizon"]:
        return out
    X, _, idx = make_windows(z, meta["seq_len"], meta["horizon"])
    pred = model(torch.from_numpy(X).to(device)).cpu().numpy() * meta["std"] + meta["mean"]
    out[idx] = pred
    return out


@torch.no_grad()
def forecast_next(model: DefectLSTM, meta: Dict[str, Any], rates: np.ndarray, device: str = "cpu") -> float:
    """Forecast the defect rate ``horizon`` batches after the end of ``rates`` (needs ``seq_len`` values)."""
    L = meta["seq_len"]
    if len(rates) < L:
        raise ValueError(f"need at least {L} batches, got {len(rates)}")
    z = (np.asarray(rates[-L:], dtype=np.float32) - meta["mean"]) / meta["std"]
    p = model(torch.from_numpy(z)[None, :, None].to(device)).item()
    return float(p * meta["std"] + meta["mean"])


# ------------------------------------------------------------------------------------------- training
def train_lstm(cfg: Optional[Dict[str, Any]] = None, csv: Optional[str] = None, epochs: Optional[int] = None,
               save: bool = True) -> Dict[str, Any]:
    """Train + evaluate the monitor. With ``save=False`` nothing is written to disk (unit tests)."""
    cfg = cfg or load_config()
    lc, sc = cfg["advanced"]["lstm"], cfg["advanced"]["stream"]
    set_seed(cfg["project"]["seed"])
    device = get_device()
    if csv:
        rates, source = load_defect_csv(csv), f"csv:{csv}"
    else:
        rates = simulate_defect_series(sc["n_batches"], sc["items_per_batch"], sc["base_rate"], cfg["project"]["seed"])["rate"]
        source = "simulated"
        logger.warning("No --csv given: using a SIMULATED defect stream (the datasets have no conveyor history)")
    L, H, thr = lc["seq_len"], lc["horizon"], lc["spike_threshold"]
    tr_end, va_end = chrono_split(len(rates))
    mean, std = float(rates[:tr_end].mean()), float(rates[:tr_end].std() + 1e-8)   # train-only scaler
    z = (rates - mean) / std
    X, y, idx = make_windows(z, L, H)
    sel = {"train": idx < tr_end, "val": (idx >= tr_end) & (idx < va_end), "test": idx >= va_end}
    if any(s.sum() == 0 for s in sel.values()):
        raise ValueError("series too short for the chosen seq_len / split")
    T = {k: (torch.from_numpy(X[m]).to(device), torch.from_numpy(y[m]).to(device)) for k, m in sel.items()}

    model = DefectLSTM(lc["hidden"], lc["layers"], lc["dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lc["lr"])
    loss_fn = nn.MSELoss()
    epochs, bs = epochs or lc["epochs"], lc["batch_size"]
    hist = {"train_loss": [], "val_loss": []}
    best, best_state = float("inf"), None
    Xtr, ytr = T["train"]
    with Timer() as t:
        for ep in range(1, epochs + 1):
            model.train()
            perm, run = torch.randperm(len(Xtr), device=device), 0.0
            for i in range(0, len(perm), bs):
                b = perm[i:i + bs]
                opt.zero_grad()
                loss = loss_fn(model(Xtr[b]), ytr[b])
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                run += loss.item() * len(b)
            model.eval()
            with torch.no_grad():
                vl = loss_fn(model(T["val"][0]), T["val"][1]).item()
            hist["train_loss"].append(run / len(Xtr))
            hist["val_loss"].append(vl)
            if vl < best:
                best, best_state = vl, {k: v.detach().clone() for k, v in model.state_dict().items()}
            logger.info("epoch %d/%d train=%.4f val=%.4f", ep, epochs, hist["train_loss"][-1], vl)
    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        pred = model(T["test"][0]).cpu().numpy() * std + mean
    actual = y[sel["test"]] * std + mean
    prev = X[sel["test"]][:, -1, 0] * std + mean                       # persistence: last observed batch
    ma = (X[sel["test"]][:, -5:, 0].mean(1)) * std + mean              # 5-batch moving average
    res = {"source": source, "n_batches": len(rates), "split": {"train": tr_end, "val": va_end - tr_end, "test": len(rates) - va_end},
           "seq_len": L, "horizon": H, "spike_threshold": thr, "epochs": epochs, "train_seconds": t.seconds,
           "lstm": _regression(actual, pred), "persistence": _regression(actual, prev), "moving_avg": _regression(actual, ma),
           "alerts_lstm": _alert_metrics(actual, pred, thr), "alerts_persistence": _alert_metrics(actual, prev, thr),
           "history": hist, "test_actual": actual.tolist(), "test_pred": pred.tolist(),
           "note": "SIMULATED stream" if source == "simulated" else "real log"}
    logger.info("test MAE lstm=%.4f persistence=%.4f moving_avg=%.4f", res["lstm"]["mae"], res["persistence"]["mae"], res["moving_avg"]["mae"])
    if save:
        _plot(res, hist, thr)
        torch.save({"state_dict": model.state_dict(), "hidden": lc["hidden"], "layers": lc["layers"], "dropout": lc["dropout"],
                    "seq_len": L, "horizon": H, "mean": mean, "std": std, "spike_threshold": thr}, checkpoint_path(cfg, "lstm_defect"))
        save_json(res, "lstm_defect", "advanced", cfg)
        import pandas as pd
        pd.DataFrame({"defect_rate": rates}).to_csv(ensure_dir(Path(cfg["paths"]["logs"]) / "advanced") / "defect_stream.csv", index=False)
    return res


def _plot(res: Dict[str, Any], hist: Dict[str, list], thr: float) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.plot(hist["train_loss"], label="train"); ax.plot(hist["val_loss"], label="val")
    ax.set(title="LSTM loss (normalised MSE)", xlabel="epoch"); ax.legend()
    save_figure(fig, "lstm_loss", "advanced")
    a, p = np.array(res["test_actual"]), np.array(res["test_pred"])
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(a, label="actual", lw=1.2); ax.plot(p, label="LSTM forecast", lw=1.2)
    ax.axhline(thr, color="red", ls="--", lw=1, label=f"spike threshold {thr}")
    ax.fill_between(range(len(a)), 0, 1, where=a >= thr, color="red", alpha=0.12, transform=ax.get_xaxis_transform())
    al = np.where(p >= thr)[0]
    ax.scatter(al, p[al], color="red", s=18, zorder=3, label="alert")
    ax.set(title=f"Defect-rate forecast on the test segment ({res['note']})", xlabel="test batch", ylabel="defect rate")
    ax.legend(ncol=4, fontsize=8)
    save_figure(fig, "lstm_forecast", "advanced")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the LSTM defect-trend monitor")
    ap.add_argument("--csv", default=None, help="real defect log (defect_rate, or defects + items columns)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    r = train_lstm(load_config(a.config), a.csv, a.epochs)
    print({k: r[k] for k in ("source", "lstm", "persistence", "moving_avg", "alerts_lstm")})


if __name__ == "__main__":
    main()
