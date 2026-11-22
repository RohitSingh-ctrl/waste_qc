"""Unit I (c): from-scratch optimizers operating in place on ``{name: ndarray}`` parameter dicts."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

ParamDict = Dict[str, np.ndarray]


class Optimizer:
    """Base class. ``step`` updates ``params`` in place from ``grads``."""

    name = "base"
    full_batch = False  # if True the trainer must use the whole dataset per update

    def __init__(self, lr: float) -> None:
        self.lr = lr
        self.t = 0

    def step(self, params: ParamDict, grads: ParamDict) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def reset(self) -> None:
        self.t = 0


class BatchGradientDescent(Optimizer):
    """theta <- theta - lr * grad, with the gradient computed over the *entire* training set."""

    name = "Batch GD"
    full_batch = True

    def step(self, params: ParamDict, grads: ParamDict) -> None:
        self.t += 1
        for k in params:
            params[k] -= self.lr * grads[k]


class SGD(Optimizer):
    """Stochastic / mini-batch gradient descent: theta <- theta - lr * grad(batch)."""

    name = "SGD"

    def step(self, params: ParamDict, grads: ParamDict) -> None:
        self.t += 1
        for k in params:
            params[k] -= self.lr * grads[k]


class Momentum(Optimizer):
    """Classical (Polyak) momentum: v <- beta*v - lr*g ; theta <- theta + v."""

    name = "Momentum"

    def __init__(self, lr: float, beta: float = 0.9) -> None:
        super().__init__(lr)
        self.beta = beta
        self.v: ParamDict = {}

    def step(self, params: ParamDict, grads: ParamDict) -> None:
        self.t += 1
        for k in params:
            v = self.v.setdefault(k, np.zeros_like(params[k]))
            v *= self.beta
            v -= self.lr * grads[k]
            params[k] += v

    def reset(self) -> None:
        super().reset()
        self.v = {}


class NAG(Momentum):
    """Nesterov Accelerated Gradient (Sutskever et al. formulation).

    Equivalent to evaluating the gradient at the look-ahead point ``theta + beta*v`` but written so the
    gradient is taken at the current parameters:
        v_new = beta*v - lr*g ;  theta <- theta - beta*v_old + (1 + beta)*v_new
    """

    name = "NAG"

    def step(self, params: ParamDict, grads: ParamDict) -> None:
        self.t += 1
        for k in params:
            v_old = self.v.setdefault(k, np.zeros_like(params[k])).copy()
            v_new = self.beta * v_old - self.lr * grads[k]
            self.v[k] = v_new
            params[k] += -self.beta * v_old + (1.0 + self.beta) * v_new


OPTIMIZERS = {"batch_gd": BatchGradientDescent, "sgd": SGD, "momentum": Momentum, "nag": NAG}


def make_optimizer(name: str, **kwargs: Any) -> Optimizer:
    """Factory: ``make_optimizer('nag', lr=0.02, beta=0.9)``."""
    return OPTIMIZERS[name](**kwargs)
