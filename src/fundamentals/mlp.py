"""Unit I (b): from-scratch Multilayer Perceptron with manual forward/backward passes.

No ``torch.nn`` and no autograd: every gradient is derived by hand and verified against finite
differences in ``tests/`` (relative error < 1e-6 in float64).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.fundamentals.activations import ACTIVATIONS, softmax
from src.fundamentals.optimizers import Optimizer
from src.utils import save_json

INITS = ("random", "xavier", "he")


def init_weights(fan_in: int, fan_out: int, scheme: str, rng: np.random.Generator) -> np.ndarray:
    """Weight initialisation: ``random`` (N(0, 0.01^2), naive), ``xavier`` (Glorot), ``he`` (Kaiming)."""
    if scheme == "random":
        std = 0.01
    elif scheme == "xavier":
        std = np.sqrt(2.0 / (fan_in + fan_out))
    elif scheme == "he":
        std = np.sqrt(2.0 / fan_in)
    else:
        raise ValueError(f"unknown init '{scheme}', choose from {INITS}")
    return rng.normal(0.0, std, size=(fan_in, fan_out))


def one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((len(y), n_classes))
    out[np.arange(len(y)), y] = 1.0
    return out


class MLP:
    """Fully connected network ``layer_sizes[0] -> ... -> layer_sizes[-1]``.

    Hidden layers use ``hidden_activation``. The output layer is softmax (``cross_entropy``) or
    identity (``mse`` against one-hot targets).

    Parameters live in ``self.params`` (``W1, b1, W2, b2, ...``); hyper-parameters (architecture,
    activation, init, loss, L2, optimizer settings) are kept separate and reported by :meth:`describe`.
    """

    def __init__(self, layer_sizes: Sequence[int], hidden_activation: str = "relu", init: str = "he",
                 loss: str = "cross_entropy", l2: float = 0.0, seed: int = 42) -> None:
        if loss not in ("cross_entropy", "mse"):
            raise ValueError("loss must be 'cross_entropy' or 'mse'")
        self.layer_sizes = list(layer_sizes)
        self.hidden_activation = hidden_activation
        self.init = init
        self.loss_name = loss
        self.l2 = l2
        self.seed = seed
        rng = np.random.default_rng(seed)
        self.params: Dict[str, np.ndarray] = {}
        for i, (a, b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:]), start=1):
            self.params[f"W{i}"] = init_weights(a, b, init, rng)
            self.params[f"b{i}"] = np.zeros(b)
        self.n_layers = len(layer_sizes) - 1
        self._act, self._act_grad = ACTIVATIONS[hidden_activation]
        self._cache: Dict[str, List[np.ndarray]] = {}

    # ------------------------------------------------------------------ forward
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass; caches pre-activations ``Z`` and activations ``A`` for backprop."""
        A, Z = [X], []
        for i in range(1, self.n_layers + 1):
            z = A[-1] @ self.params[f"W{i}"] + self.params[f"b{i}"]
            Z.append(z)
            if i < self.n_layers:
                A.append(self._act(z))
            else:
                A.append(softmax(z) if self.loss_name == "cross_entropy" else z)
        self._cache = {"A": A, "Z": Z}
        return A[-1]

    # ------------------------------------------------------------------ loss
    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Mean data loss + 0.5*l2*sum(W^2)."""
        out = self.forward(X)
        Y = one_hot(y, self.layer_sizes[-1])
        if self.loss_name == "cross_entropy":
            data = -np.mean(np.log(np.clip(out[np.arange(len(y)), y], 1e-12, None)))
        else:
            data = 0.5 * np.sum((out - Y) ** 2) / len(y)
        reg = 0.5 * self.l2 * sum(float(np.sum(self.params[f"W{i}"] ** 2)) for i in range(1, self.n_layers + 1))
        return float(data + reg)

    # ------------------------------------------------------------------ backward
    def backward(self, y: np.ndarray) -> Dict[str, np.ndarray]:
        """Manual backpropagation using the cache from the last :meth:`forward` call.

        Output layer: ``dZ_L = (P - Y)/N`` for softmax+CE and ``(O - Y)/N`` for MSE (identity output).
        Hidden layers: ``dA = dZ_{l+1} W_{l+1}^T``, ``dZ_l = dA * f'(Z_l)``.
        ``dW_l = A_{l-1}^T dZ_l (+ l2 * W_l)``, ``db_l = sum_n dZ_l``.
        """
        A, Z = self._cache["A"], self._cache["Z"]
        N = len(y)
        dZ = (A[-1] - one_hot(y, self.layer_sizes[-1])) / N
        grads: Dict[str, np.ndarray] = {}
        for i in range(self.n_layers, 0, -1):
            grads[f"W{i}"] = A[i - 1].T @ dZ + self.l2 * self.params[f"W{i}"]
            grads[f"b{i}"] = dZ.sum(axis=0)
            if i > 1:
                dA = dZ @ self.params[f"W{i}"].T
                dZ = dA * self._act_grad(Z[i - 2], A[i - 1])
        return grads

    def loss_and_grads(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, Dict[str, np.ndarray]]:
        """Convenience: forward + loss + backward for one batch."""
        loss = self.loss(X, y)
        return loss, self.backward(y)

    # ------------------------------------------------------------------ inference
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.forward(X), axis=1)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y))

    # ------------------------------------------------------------------ introspection
    def n_parameters(self) -> int:
        return int(sum(p.size for p in self.params.values()))

    def describe(self, optimizer: Optional[Optimizer] = None, batch_size: Optional[int] = None,
                 epochs: Optional[int] = None) -> Dict[str, Any]:
        """Separate *parameters* (learned) from *hyper-parameters* (chosen) for logging."""
        hyper: Dict[str, Any] = {"layer_sizes": self.layer_sizes, "hidden_activation": self.hidden_activation,
                                 "init": self.init, "loss": self.loss_name, "l2": self.l2, "seed": self.seed}
        if optimizer is not None:
            hyper["optimizer"] = {"name": optimizer.name, "lr": optimizer.lr,
                                  **({"beta": optimizer.beta} if hasattr(optimizer, "beta") else {})}
        if batch_size is not None:
            hyper["batch_size"] = batch_size
        if epochs is not None:
            hyper["epochs"] = epochs
        params = {k: {"shape": list(v.shape), "count": int(v.size), "mean": float(v.mean()),
                      "std": float(v.std()), "l2_norm": float(np.linalg.norm(v))} for k, v in self.params.items()}
        return {"hyperparameters": hyper, "parameters": params, "total_parameters": self.n_parameters()}


@dataclass
class TrainingLogger:
    """Per-epoch metrics plus the hyper-parameter/parameter description; persisted as JSON."""

    description: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, float]] = field(default_factory=list)

    def log(self, **metrics: float) -> None:
        self.history.append({k: float(v) for k, v in metrics.items()})

    def column(self, key: str) -> np.ndarray:
        return np.asarray([h[key] for h in self.history if key in h])

    def save(self, name: str, subdir: str = "fundamentals") -> str:
        return str(save_json({"description": self.description, "history": self.history}, name, subdir))


def fit(model: MLP, X: np.ndarray, y: np.ndarray, optimizer: Optimizer, epochs: int = 50, batch_size: int = 32,
        X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None, seed: int = 42,
        shuffle: bool = True) -> TrainingLogger:
    """Train ``model`` with a from-scratch ``optimizer``.

    Batch GD (``optimizer.full_batch``) forces ``batch_size = N``. Train loss/accuracy are re-measured on
    the full training set after every epoch so that optimizers are compared on the same footing.
    """
    rng = np.random.default_rng(seed)
    N = len(X)
    bs = N if optimizer.full_batch else max(1, min(batch_size, N))
    lg = TrainingLogger(description=model.describe(optimizer, bs, epochs))
    n_updates, t0 = 0, time.perf_counter()
    for ep in range(1, epochs + 1):
        order = rng.permutation(N) if shuffle else np.arange(N)
        for s in range(0, N, bs):
            idx = order[s:s + bs]
            _, grads = model.loss_and_grads(X[idx], y[idx])
            optimizer.step(model.params, grads)
            n_updates += 1
        rec = {"epoch": ep, "updates": n_updates, "train_loss": model.loss(X, y), "train_acc": model.accuracy(X, y),
               "time_s": time.perf_counter() - t0}
        if X_val is not None and y_val is not None:
            rec.update(val_loss=model.loss(X_val, y_val), val_acc=model.accuracy(X_val, y_val))
        lg.log(**rec)
    return lg
