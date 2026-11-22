"""Unit I (a): McCulloch-Pitts and Sigmoid neurons built from scratch with NumPy."""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from src.fundamentals.activations import sigmoid
from src.utils import get_logger, save_figure, save_json, set_seed

logger = get_logger(__name__)


class McCullochPittsNeuron:
    """Binary threshold unit (McCulloch & Pitts, 1943).

    Fires (1) iff no inhibitory input is active and the sum of active excitatory inputs ``>= threshold``.
    """

    def __init__(self, n_inputs: int, threshold: int, inhibitory: Optional[Sequence[bool]] = None) -> None:
        self.n_inputs = n_inputs
        self.threshold = threshold
        self.inhibitory = np.zeros(n_inputs, dtype=bool) if inhibitory is None else np.asarray(inhibitory, dtype=bool)

    def __call__(self, x: Sequence[int]) -> int:
        x = np.asarray(x, dtype=int)
        if x.shape != (self.n_inputs,):
            raise ValueError(f"expected {self.n_inputs} inputs, got {x.shape}")
        if np.any(x[self.inhibitory] > 0):  # absolute inhibition
            return 0
        return int(x[~self.inhibitory].sum() >= self.threshold)


def and_gate(n: int = 2) -> McCullochPittsNeuron:
    return McCullochPittsNeuron(n, threshold=n)


def or_gate(n: int = 2) -> McCullochPittsNeuron:
    return McCullochPittsNeuron(n, threshold=1)


def not_gate() -> Callable[[int], int]:
    """NOT via an always-on bias input and one inhibitory input."""
    neuron = McCullochPittsNeuron(2, threshold=1, inhibitory=[True, False])
    return lambda x: neuron([x, 1])


def nand_gate() -> Callable[[int, int], int]:
    a, n = and_gate(), not_gate()
    return lambda x, y: n(a([x, y]))


def nor_gate() -> Callable[[int, int], int]:
    o, n = or_gate(), not_gate()
    return lambda x, y: n(o([x, y]))


def xor_gate() -> Callable[[int, int], int]:
    """XOR needs two layers of MP neurons: AND(OR(x,y), NAND(x,y))."""
    o, nd, a = or_gate(), nand_gate(), and_gate()
    return lambda x, y: a([o([x, y]), nd(x, y)])


def truth_table(fn: Callable[..., int], n_inputs: int = 2) -> Dict[tuple, int]:
    """Enumerate a gate's truth table."""
    from itertools import product

    return {bits: int(fn(*bits)) for bits in product([0, 1], repeat=n_inputs)}


def contamination_rule(hazardous: int, non_recyclable: int, wet: int) -> int:
    """Domain demo: flag a bale if hazardous OR (non-recyclable AND wet), built from MP gates."""
    return or_gate()([hazardous, and_gate()([non_recyclable, wet])])


class SigmoidNeuron:
    """Single logistic neuron ``a = sigmoid(w.x + b)`` trained with gradient descent on binary cross-entropy."""

    def __init__(self, n_inputs: int, lr: float = 0.5, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0, 0.5, size=n_inputs)
        self.b = 0.0
        self.lr = lr

    def forward(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(X @ self.w + self.b)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.forward(X) >= 0.5).astype(int)

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        a = np.clip(self.forward(X), 1e-12, 1 - 1e-12)
        return float(-np.mean(y * np.log(a) + (1 - y) * np.log(1 - a)))

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 500) -> List[float]:
        """Full-batch gradient descent; returns the loss history. dL/dz = (a - y)/N."""
        hist = []
        for _ in range(epochs):
            dz = (self.forward(X) - y) / len(y)
            self.w -= self.lr * (X.T @ dz)
            self.b -= self.lr * dz.sum()
            hist.append(self.loss(X, y))
        return hist


def demo(seed: int = 42) -> Dict[str, object]:
    """Truth tables for MP gates + sigmoid neuron on AND/OR/XOR (XOR fails: not linearly separable)."""
    set_seed(seed)
    gates = {"AND": and_gate(), "OR": or_gate()}
    tables = {k: truth_table(lambda x, y, g=g: g([x, y])) for k, g in gates.items()}
    tables.update({"NOT": {(0,): not_gate()(0), (1,): not_gate()(1)}, "NAND": truth_table(nand_gate()),
                   "NOR": truth_table(nor_gate()), "XOR(2-layer)": truth_table(xor_gate())})
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    targets = {"AND": [0, 0, 0, 1], "OR": [0, 1, 1, 1], "XOR": [0, 1, 1, 0]}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    accs: Dict[str, float] = {}
    gx, gy = np.meshgrid(np.linspace(-0.5, 1.5, 100), np.linspace(-0.5, 1.5, 100))
    grid = np.c_[gx.ravel(), gy.ravel()]
    for ax, (name, t) in zip(axes, targets.items()):
        y = np.asarray(t, dtype=float)
        neuron = SigmoidNeuron(2, lr=1.0, seed=seed)
        neuron.fit(X, y, epochs=3000)
        accs[name] = float((neuron.predict(X) == y).mean())
        ax.contourf(gx, gy, neuron.forward(grid).reshape(gx.shape), levels=20, cmap="RdBu_r", alpha=0.6)
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="k", s=120)
        ax.set_title(f"Sigmoid neuron: {name} (acc={accs[name]:.2f})")
    fig_path = save_figure(fig, "neurons_decision_boundaries", subdir="fundamentals")
    out = {"mp_truth_tables": {k: {str(b): v for b, v in tb.items()} for k, tb in tables.items()},
           "sigmoid_accuracy": accs,
           "contamination_rule": {str(b): contamination_rule(*b) for b in [(0, 0, 0), (1, 0, 0), (0, 1, 1), (0, 1, 0)]},
           "figure": str(fig_path)}
    save_json(out, "neurons_demo")
    return out


if __name__ == "__main__":
    import json

    print(json.dumps(demo(), indent=2))
