"""Unit IV (c): object detection - Faster R-CNN (torchvision) and YOLOv8 (ultralytics), benchmarked with one mAP."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from src.evaluation.benchmark import count_parameters, measure_latency, model_size_mb
from src.evaluation.metrics import detection_map
from src.utils import ensure_dir, get_logger, load_config, save_json, set_seed
from src.vision.common import Timer, checkpoint_path, detection_tag, get_device, parse_common_args, processed_dir
from src.vision.datasets import YoloFolderDetection, detection_collate

logger = get_logger(__name__)


def build_faster_rcnn(n_classes: int, pretrained: bool = False, min_size: int = 320) -> torch.nn.Module:
    """Faster R-CNN ResNet-50-FPN with a new box predictor for ``n_classes`` (+ background)."""
    m = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None,
                                weights_backbone=None, min_size=min_size, max_size=min_size * 2)
    m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, n_classes + 1)
    return m


@torch.no_grad()
def run_detector(model: torch.nn.Module, loader: DataLoader, device: torch.device, conf: float):
    """Returns (preds, gts) in the 0-based format expected by :func:`detection_map`."""
    model.eval()
    preds: List[Dict[str, np.ndarray]] = []
    gts: List[Dict[str, np.ndarray]] = []
    for imgs, tgts in loader:
        outs = model([i.to(device) for i in imgs])
        for o, t in zip(outs, tgts):
            keep = o["scores"] >= conf
            preds.append({"boxes": o["boxes"][keep].cpu().numpy(), "scores": o["scores"][keep].cpu().numpy(),
                          "labels": o["labels"][keep].cpu().numpy() - 1})
            gts.append({"boxes": t["boxes"].numpy(), "labels": t["labels"].numpy() - 1})
    return preds, gts


def train_faster_rcnn(cfg: Optional[Dict[str, Any]] = None, synthetic: bool = False, epochs: Optional[int] = None,
                      image_size: int = 320) -> Dict[str, Any]:
    cfg = cfg or load_config()
    vc = cfg["vision"]["detection"]
    set_seed(cfg["project"]["seed"])
    device, n = get_device(), len(cfg["classes"]["names"])
    root = processed_dir(cfg, detection_tag(cfg, synthetic), "yolo")
    dl = {s: DataLoader(YoloFolderDetection(root, s), vc["batch_size"], shuffle=(s == "train"),
                        collate_fn=detection_collate) for s in ("train", "val", "test")}
    model = build_faster_rcnn(n, cfg["vision"]["classification"]["pretrained"], image_size).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=5e-4)
    epochs, hist = epochs or vc["epochs"], []
    with Timer() as t:
        for ep in range(1, epochs + 1):
            model.train()
            run, k = 0.0, 0
            for imgs, tgts in dl["train"]:
                imgs = [i.to(device) for i in imgs]
                tgts = [{k_: v.to(device) for k_, v in tg.items()} for tg in tgts]
                loss = sum(model(imgs, tgts).values())
                opt.zero_grad()
                loss.backward()
                opt.step()
                run, k = run + loss.item(), k + 1
            hist.append(run / max(k, 1))
            logger.info("epoch %d/%d loss=%.4f", ep, epochs, hist[-1])
    preds, gts = run_detector(model, dl["test"], device, vc["conf"])
    res = detection_map(preds, gts, n)
    sample = [next(iter(dl["test"]))[0][0].to(device)]
    with torch.no_grad():
        lat = measure_latency(lambda: model(sample), warmup=1, runs=5)
    out = {"model": "fasterrcnn", "epochs": epochs, "train_seconds": t.seconds, "params": count_parameters(model),
           "size_mb": model_size_mb(model), **lat, "train_loss": hist, **res}
    torch.save({"model": "fasterrcnn", "state_dict": model.state_dict(), "n_classes": n}, checkpoint_path(cfg, "fasterrcnn"))
    save_json(out, "detection_fasterrcnn", "vision")
    logger.info("Faster R-CNN mAP50=%.3f mAP50-95=%.3f", res["map50"], res["map50_95"])
    return out


def train_yolov8(cfg: Optional[Dict[str, Any]] = None, synthetic: bool = False, epochs: Optional[int] = None,
                 image_size: int = 320, weights: str = "yolov8n.pt") -> Dict[str, Any]:
    """Fine-tune YOLOv8 with ultralytics. ``weights='yolov8n.yaml'`` trains from scratch (no download)."""
    from ultralytics import YOLO

    cfg = cfg or load_config()
    vc = cfg["vision"]["detection"]
    root = processed_dir(cfg, detection_tag(cfg, synthetic), "yolo").resolve()
    # data.yaml stores an absolute path from the machine that ran the pipeline -> rewrite for *this* machine (Drive/Colab).
    d = yaml.safe_load((root / "data.yaml").read_text())
    d["path"] = str(root)
    runtime_yaml = root / "data_runtime.yaml"
    runtime_yaml.write_text(yaml.safe_dump(d))
    runs = ensure_dir(cfg["paths"]["outputs"]) / "yolo_runs"
    model = YOLO(weights)
    epochs = epochs or vc["epochs"]
    with Timer() as t:
        model.train(data=str(runtime_yaml), epochs=epochs, imgsz=image_size, batch=vc["batch_size"], project=str(runs),
                    name="yolov8", exist_ok=True, seed=cfg["project"]["seed"], plots=False, verbose=False,
                    device=0 if torch.cuda.is_available() else "cpu", workers=0)
    m = model.val(data=str(runtime_yaml), split="test", imgsz=image_size, conf=0.001, iou=vc["iou"], plots=False,
                  verbose=False, workers=0, project=str(runs), name="yolov8_test", exist_ok=True)
    img = next(iter((root / "images" / "test").glob("*.*")))
    lat = measure_latency(lambda: model.predict(str(img), imgsz=image_size, conf=vc["conf"], verbose=False), warmup=2, runs=10)
    out = {"model": "yolov8", "weights": weights, "epochs": epochs, "train_seconds": t.seconds,
           "map50": float(m.box.map50), "map50_95": float(m.box.map), **lat}
    best = runs / "yolov8" / "weights" / "best.pt"
    if best.exists():
        shutil.copy2(best, checkpoint_path(cfg, "yolov8"))
    save_json(out, "detection_yolov8", "vision")
    logger.info("YOLOv8 mAP50=%.3f mAP50-95=%.3f", out["map50"], out["map50_95"])
    return out


def main() -> None:
    ap = parse_common_args("Train detectors")
    ap.add_argument("--model", choices=["fasterrcnn", "yolov8", "all"], default="all")
    ap.add_argument("--yolo-weights", default="yolov8n.pt", help="use yolov8n.yaml to train without downloading weights")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.model in ("fasterrcnn", "all"):
        train_faster_rcnn(cfg, a.synthetic, a.epochs, a.image_size or 320)
    if a.model in ("yolov8", "all"):
        train_yolov8(cfg, a.synthetic, a.epochs, a.image_size or 320, a.yolo_weights)


if __name__ == "__main__":
    main()
