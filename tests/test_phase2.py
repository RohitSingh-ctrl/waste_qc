"""Phase 2 tests: metrics, model shapes, convolution correctness, learning-rule behaviour."""
import numpy as np
import pytest
import torch

from src.evaluation.metrics import (classification_report, detection_map, recycling_purity, segmentation_metrics,
                                    box_iou_matrix)
from src.experiments.conv_visualization import verify_against_torch
from src.experiments.learning_rules import bars_and_stripes, oja_demo, error_correction_demo, hebbian_demo, rbm_demo
from src.vision.classification import build_classifier
from src.vision.detection import build_faster_rcnn
from src.vision.unet import UNet, dice_loss

NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


def test_classification_report_perfect_and_purity():
    y = np.array([0, 1, 2, 3, 4, 5])
    r = classification_report(y, y, NAMES)
    assert r["accuracy"] == 1.0 and r["macro_f1"] == 1.0
    p = recycling_purity([5, 5, 0, 1], [4, 5, 0, 1], NAMES, NAMES[:5])  # one trash item sorted as plastic
    assert p["purity_rate"] == pytest.approx(2 / 3) and p["recovery_rate"] == 1.0


def test_segmentation_metrics():
    t = np.zeros((2, 8, 8), int); t[:, :4] = 1
    assert segmentation_metrics(t, t, 3)["miou"] == 1.0
    assert segmentation_metrics(np.zeros_like(t), t, 3)["miou"] < 1.0


def test_detection_map_perfect_and_empty():
    box = np.array([[10, 10, 50, 50]], float)
    gts = [{"boxes": box, "labels": np.array([1])}]
    good = [{"boxes": box, "scores": np.array([0.9]), "labels": np.array([1])}]
    bad = [{"boxes": np.zeros((0, 4)), "scores": np.zeros(0), "labels": np.zeros(0, int)}]
    assert detection_map(good, gts, 6)["map50"] == pytest.approx(1.0)
    assert detection_map(bad, gts, 6)["map50"] == 0.0
    assert box_iou_matrix(box, box)[0, 0] == pytest.approx(1.0)


def test_unet_shapes_and_dice():
    m = UNet(7, base=8, depth=3)
    x = torch.randn(2, 3, 32, 32)
    out = m(x)
    assert out.shape == (2, 7, 32, 32)
    assert 0.0 <= float(dice_loss(out.detach(), torch.randint(0, 7, (2, 32, 32)))) <= 1.0


@pytest.mark.parametrize("name", ["resnet50", "googlenet"])
def test_classifier_heads(name):
    m = build_classifier(name, 6).eval()
    assert m(torch.randn(2, 3, 64, 64)).shape == (2, 6)


def test_faster_rcnn_builds_and_infers():
    m = build_faster_rcnn(6, min_size=96).eval()
    with torch.no_grad():
        out = m([torch.rand(3, 96, 96)])
    assert set(out[0]) >= {"boxes", "scores", "labels"}


def test_manual_convolution_matches_torch():
    d = verify_against_torch()
    assert d["conv2d_max_abs_diff"] < 1e-9 and d["conv3d_max_abs_diff"] < 1e-9


def test_learning_rules_behave_as_theory_predicts():
    assert hebbian_demo()["norms"][-1] > 100 * hebbian_demo()["norms"][0]   # plain Hebb diverges
    o = oja_demo()
    assert abs(o["norms"][-1] - 1.0) < 0.05 and o["cosine"] > 0.99           # Oja -> unit-norm PC1
    assert error_correction_demo()["errors"][-1] == 0                        # separable -> converges
    assert rbm_demo()["errors"][-1] < 0.01 and len(bars_and_stripes(4)) == 30
