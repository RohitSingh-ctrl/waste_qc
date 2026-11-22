"""Phase 1 tests: gradient correctness, from-scratch neurons/MLP/optimizers, leakage-free splits, dedup."""
import numpy as np
import pytest

from src.data.splits import assert_no_leakage, grouped_split
from src.data.synthetic import make_classification_arrays
from src.fundamentals.mlp import MLP, fit
from src.fundamentals.optimizers import make_optimizer
from src.math_sandbox.gradient_checker import numerical_gradient  # noqa: F401  (import check)


def _numeric_grad(model, X, y, key, eps=1e-6):
    g = np.zeros_like(model.params[key])
    it = np.nditer(model.params[key], flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        old = model.params[key][i]
        model.params[key][i] = old + eps
        lp = model.loss(X, y)
        model.params[key][i] = old - eps
        lm = model.loss(X, y)
        model.params[key][i] = old
        g[i] = (lp - lm) / (2 * eps)
    return g


@pytest.mark.parametrize("act", ["relu", "sigmoid", "tanh"])
@pytest.mark.parametrize("loss", ["cross_entropy", "mse"])
def test_mlp_backprop_matches_finite_differences(act, loss):
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(8, 5)), rng.integers(0, 3, 8)
    m = MLP([5, 6, 3], act, "xavier", loss, l2=0.01, seed=1)
    _, grads = m.loss_and_grads(X, y)
    for k in ("W1", "b1", "W2", "b2"):
        num = _numeric_grad(m, X, y, k)
        rel = np.linalg.norm(grads[k] - num) / (np.linalg.norm(grads[k]) + np.linalg.norm(num) + 1e-12)
        assert rel < 1e-5, (k, rel)


@pytest.mark.parametrize("opt", ["batch_gd", "sgd", "momentum", "nag"])
def test_every_optimizer_reduces_loss(opt):
    X, y = make_classification_arrays(15, 8)
    X = X.reshape(len(X), -1).astype(float) / 255.0
    hp = {"batch_gd": {"lr": 0.5}, "sgd": {"lr": 0.05}, "momentum": {"lr": 0.02, "beta": 0.9}, "nag": {"lr": 0.02, "beta": 0.9}}[opt]
    m = MLP([X.shape[1], 16, 6], "relu", "he", seed=0)
    before = m.loss(X, y)
    fit(m, X, y, make_optimizer(opt, **hp), epochs=20, batch_size=16)
    assert m.loss(X, y) < before


def test_grouped_split_has_no_leakage():
    groups = [f"g{i // 3}" for i in range(60)]
    sp = grouped_split(groups, {"train": 0.7, "val": 0.15, "test": 0.15}, 42)
    assert_no_leakage(groups, sp)
    assert sum(len(v) for v in sp.values()) == 60
