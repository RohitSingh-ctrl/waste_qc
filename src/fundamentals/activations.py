"""Activation functions and derivatives implemented with raw NumPy (shared by neurons, MLP, Unit V study)."""
from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np


def step(z: np.ndarray) -> np.ndarray:
    """Heaviside step (1 if z >= 0). Derivative is 0 almost everywhere -> cannot be trained by gradients."""
    return (z >= 0).astype(np.float64)


def step_grad(z: np.ndarray, a: np.ndarray) -> np.ndarray:
    return np.zeros_like(z)


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid."""
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def sigmoid_grad(z: np.ndarray, a: np.ndarray) -> np.ndarray:
    return a * (1.0 - a)


def tanh(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)


def tanh_grad(z: np.ndarray, a: np.ndarray) -> np.ndarray:
    return 1.0 - a ** 2


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(z, 0.0)


def relu_grad(z: np.ndarray, a: np.ndarray) -> np.ndarray:
    return (z > 0).astype(np.float64)


def identity(z: np.ndarray) -> np.ndarray:
    return z


def identity_grad(z: np.ndarray, a: np.ndarray) -> np.ndarray:
    return np.ones_like(z)


def softmax(z: np.ndarray) -> np.ndarray:
    """Row-wise stable softmax."""
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# name -> (forward(z), grad(z, a)) where a = forward(z)
ACTIVATIONS: Dict[str, Tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray, np.ndarray], np.ndarray]]] = {
    "step": (step, step_grad),
    "sigmoid": (sigmoid, sigmoid_grad),
    "tanh": (tanh, tanh_grad),
    "relu": (relu, relu_grad),
    "identity": (identity, identity_grad),
}
