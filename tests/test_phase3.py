"""Phase 3 tests: LSTM trend monitor, VAE (+ step/sigmoid/tanh), robustness, deployment check, GUI helpers."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from app.inference import batch_summary, list_classifiers, load_classifier, predict_image, route_item
from src.advanced.lstm_monitor import (LSTMTrend, chronological_split, class_priors, make_windows, run_lstm_monitor,
                                       simulate_stream, spike_metrics)
from src.advanced.vae import (ConvVAE, Step, make_activation, run_activation_comparison, run_generate_rare,
                              run_latent_2d, vae_loss)
from src.evaluation.deployment import evaluate_deployment, evaluate_model, render_report, run_deployment_check
from src.evaluation.robustness import STRESSES, apply_stress, evaluate_robustness_on
from src.utils import load_config

NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


@pytest.fixture()
def cfg(tmp_path):
    c = load_config()
    for k in ("outputs", "figures", "logs", "checkpoints"):
        c["paths"][k] = str(tmp_path / k)
    return c


# ----------------------------------------------------------------------------- LSTM monitor
def test_stream_is_reproducible_and_has_episodes():
    pri = np.array([0.2, 0.2, 0.2, 0.2, 0.15, 0.05])
    a, b = simulate_stream(pri, [5], 600, seed=1), simulate_stream(pri, [5], 600, seed=1)
    assert np.array_equal(a["fractions"], b["fractions"])
    assert np.allclose(a["fractions"].sum(1), 1.0) and a["episode"].any()
    assert a["rate"][a["episode"]].mean() > a["rate"][~a["episode"]].mean()


def test_windows_and_chronological_split_have_no_leakage():
    pri = np.full(6, 1 / 6)
    s = simulate_stream(pri, [5], 400, seed=0)
    X, y = make_windows(s["fractions"], s["rate"], 10)
    assert X.shape == (390, 10, 6) and np.allclose(y, s["rate"][10:])
    d = chronological_split(X, y, 10)
    assert sum(len(v[0]) for v in d.values()) < len(X)           # purge gaps removed
    # first val window starts after the last train *target*
    assert (len(d["train"][0]) + 10) <= (len(X) - len(d["val"][0]) - len(d["test"][0]) - 10 + 1)


def test_lstm_forward_range_and_spike_metrics():
    m = LSTMTrend(6, hidden=8, layers=2)
    out = m(torch.rand(4, 10, 6))
    assert out.shape == (4,) and (out >= 0).all() and (out <= 1).all()
    r = spike_metrics(np.array([0.1, 0.5, 0.6, 0.1]), np.array([0.1, 0.5, 0.1, 0.5]), 0.35)
    assert (r["tp"], r["fp"], r["fn"]) == (1, 1, 1) and r["f1"] == pytest.approx(0.5)


def test_run_lstm_monitor_end_to_end(cfg):
    r = run_lstm_monitor(cfg, synthetic=True, epochs=2, n_steps=600)
    assert r["simulated_stream"] is True and set(r["test"]) == {"lstm", "persistence", "moving_avg_5"}
    assert Path(r["figure"]).exists() and (Path(cfg["paths"]["checkpoints"]) / "lstm_monitor.pt").exists()
    assert abs(sum(class_priors(cfg, True)) - 1) < 1e-9


# ----------------------------------------------------------------------------- VAE
def test_step_activation_forward_and_straight_through_gradient():
    x = torch.tensor([-2.0, -0.5, 0.5, 2.0], requires_grad=True)
    y = Step()(x)
    assert y.tolist() == [0.0, 0.0, 1.0, 1.0]
    y.sum().backward()
    assert x.grad.tolist() == [0.0, 1.0, 1.0, 0.0]
    with pytest.raises(ValueError):
        make_activation("nope")


@pytest.mark.parametrize("act", ["step", "sigmoid", "tanh"])
def test_vae_shapes_and_loss(act):
    m = ConvVAE(latent_dim=4, size=16, hidden_activation=act)
    x = torch.rand(3, 3, 16, 16)
    rec, mu, lv = m(x)
    assert rec.shape == x.shape and mu.shape == (3, 4)
    loss, r, kl = vae_loss(rec, x, mu, lv)
    assert torch.isfinite(loss) and float(kl) >= 0
    loss.backward()
    assert all(p.grad is not None for p in m.fc_mu.parameters())


def test_vae_rejects_bad_size():
    with pytest.raises(ValueError):
        ConvVAE(size=20)


def test_vae_experiments_run_on_synthetic(cfg):
    kw = dict(synthetic=True, epochs=1, size=16, per_class=8)
    c = run_activation_comparison(cfg, **kw)
    assert set(c["results"]) == {"step", "sigmoid", "tanh"} and Path(c["figure"]).exists()
    l = run_latent_2d(cfg, **kw)
    assert Path(l["figure"]).exists() and all(len(v) == 2 for v in l["class_centroids"].values())
    g = run_generate_rare(cfg, **kw)
    assert g["class"] == "trash" and Path(g["figure"]).exists()


# ----------------------------------------------------------------------------- robustness
def _items(n=12, size=32, seed=0):
    rng = np.random.default_rng(seed)
    return [(Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8)), int(i % 6)) for i in range(n)]


def test_stresses_keep_size_and_change_pixels():
    img = _items(1, 64)[0][0]
    for s in STRESSES:
        out = apply_stress(img, s)
        assert out.size == img.size
        assert np.abs(np.asarray(out, float) - np.asarray(img, float)).mean() > 0


def test_evaluate_robustness_structure():
    model = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 6))
    r = evaluate_robustness_on(model, _items(), NAMES, size=32, device=torch.device("cpu"))
    assert set(r["stress"]) == set(STRESSES) and r["n_images"] == 12
    assert r["worst_stress"] in STRESSES and r["max_relative_drop"] >= max(v["relative_drop"] for v in r["stress"].values()) - 1e-12


# ----------------------------------------------------------------------------- deployment check
GOOD_CLS = {"test": {"purity_rate": 0.97, "macro_f1": 0.9}, "fps": 50, "latency_ms_mean": 20, "size_mb": 100,
            "robustness": {"max_relative_drop": 0.05}}
GOOD_DET = {"map50": 0.8, "map50_95": 0.5, "fps": 30, "latency_ms_mean": 30, "size_mb": 50}


def test_evaluate_model_pass_fail_and_missing():
    thr = load_config()["deployment"]
    assert evaluate_model(GOOD_CLS, "classification", thr)["verdict"] == "READY"
    bad = {**GOOD_CLS, "test": {"purity_rate": 0.80, "macro_f1": 0.9}}
    assert evaluate_model(bad, "classification", thr)["verdict"] == "FAIL"
    no_rob = {k: v for k, v in GOOD_CLS.items() if k != "robustness"}
    m = evaluate_model(no_rob, "classification", thr)
    assert m["verdict"] == "INCOMPLETE" and any(r["status"] == "not_measured" for r in m["rows"])


def test_system_verdicts():
    thr = load_config()["deployment"]
    ready = evaluate_deployment({"classification": {"resnet50": GOOD_CLS}, "detection": {"yolov8": GOOD_DET}}, thr)
    assert ready["system_verdict"] == "READY" and ready["best_per_task"]["detection"] == "detection/yolov8"
    assert evaluate_deployment({"classification": {}, "detection": {}}, thr)["system_verdict"] == "INCOMPLETE"
    only_cls = evaluate_deployment({"classification": {"resnet50": GOOD_CLS}, "detection": {}}, thr)
    assert only_cls["system_verdict"] == "INCOMPLETE"
    weak = {**GOOD_DET, "map50": 0.1}
    assert evaluate_deployment({"classification": {"resnet50": GOOD_CLS}, "detection": {"yolov8": weak}}, thr)["system_verdict"] == "NOT READY"
    assert "System verdict" in render_report(ready)


def test_run_deployment_check_reads_saved_json(cfg):
    d = Path(cfg["paths"]["logs"]) / "vision"
    d.mkdir(parents=True)
    (d / "classification_resnet50.json").write_text(json.dumps(GOOD_CLS | {"model": "resnet50"}))
    (d / "robustness_resnet50.json").write_text(json.dumps({"max_relative_drop": 0.05}))
    (d / "detection_yolov8.json").write_text(json.dumps(GOOD_DET))
    out = run_deployment_check(cfg)
    assert out["system_verdict"] == "READY" and Path(out["report"]).exists()


# ----------------------------------------------------------------------------- GUI helpers
def test_route_item_and_batch_summary(cfg):
    assert route_item({"plastic": 0.9, "trash": 0.1}, cfg)["action"] == "recycle"
    assert route_item({"plastic": 0.1, "trash": 0.9}, cfg)["action"] == "reject"
    assert route_item({"plastic": 0.4, "trash": 0.35}, cfg, 0.5)["action"] == "manual check"
    s = batch_summary([{"label": "plastic", "action": "recycle"}, {"label": "trash", "action": "reject"}])
    assert s["n"] == 2 and s["predicted_reject_share"] == 0.5 and s["class_counts"]["trash"] == 1
    assert batch_summary([])["predicted_reject_share"] == 0.0


def test_load_classifier_roundtrip(cfg):
    from src.vision.classification import build_classifier
    from src.vision.common import checkpoint_path
    assert list_classifiers(cfg) == []
    torch.save({"model": "googlenet", "state_dict": build_classifier("googlenet", 6).state_dict(), "classes": NAMES},
               checkpoint_path(cfg, "googlenet"))
    assert list_classifiers(cfg) == ["googlenet"]
    model, names = load_classifier("googlenet", cfg)
    probs = predict_image(model, _items(1, 48)[0][0], names, size=64)
    assert names == NAMES and sum(probs.values()) == pytest.approx(1.0, abs=1e-4)
    with pytest.raises(FileNotFoundError):
        load_classifier("resnet50", cfg)


def test_gui_script_runs_without_checkpoints():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app" / "gui.py"), default_timeout=60).run()
    assert not at.exception
