"""Deployment readiness check: compares measured / logged metrics with ``deployment:`` thresholds in the config.

Rules that keep the verdict honest:
* a metric that could not be evaluated is ``NOT_EVALUATED`` and counts as *not ready* (never a silent pass)
* classifiers are benchmarked on THIS machine (latency, FPS, memory) and stress-tested (noise, blur, brightness, rotation)
* detector numbers (mAP, FPS) are read from the Phase 2 logs, which were measured where the detector was trained
Exit code is 0 only when the required path(s) pass, so it can be used in CI.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from src.evaluation.benchmark import measure_latency, peak_gpu_memory_mb
from src.evaluation.metrics import classification_report, recycling_purity
from src.utils import ensure_dir, get_logger, load_config, resolve_path, save_json, set_seed

logger = get_logger(__name__)
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
PASS, FAIL, NOT_EVAL = "PASS", "FAIL", "NOT_EVALUATED"

# (metric key, threshold key in config.deployment, operator)
CLASSIFIER_CHECKS = [("macro_f1", "min_macro_f1", ">="), ("purity_rate", "min_purity_rate", ">="), ("fps", "min_fps", ">="),
                     ("latency_ms", "max_latency_ms", "<="), ("memory_mb", "max_memory_mb", "<="),
                     ("robustness_drop", "max_robustness_drop", "<=")]
DETECTOR_CHECKS = [("map50", "min_map50", ">="), ("map50_95", "min_map50_95", ">="), ("fps", "min_fps", ">="),
                   ("latency_ms", "max_latency_ms", "<=")]


# ------------------------------------------------------------------------------------------- pure logic
def check_metrics(metrics: Dict[str, Optional[float]], thresholds: Dict[str, float],
                  checks: Sequence[Tuple[str, str, str]]) -> List[Dict[str, Any]]:
    """Compare ``metrics`` with ``thresholds``. Missing / NaN metrics -> NOT_EVALUATED."""
    rows = []
    for key, tkey, op in checks:
        v, t = metrics.get(key), thresholds[tkey]
        if v is None or (isinstance(v, float) and np.isnan(v)):
            status = NOT_EVAL
        else:
            status = PASS if (v >= t if op == ">=" else v <= t) else FAIL
        rows.append({"metric": key, "value": v, "op": op, "threshold": t, "status": status})
    return rows


def model_ready(rows: Sequence[Dict[str, Any]]) -> bool:
    return bool(rows) and all(r["status"] == PASS for r in rows)


def overall_verdict(models: Dict[str, Dict[str, Any]], require: str = "both") -> Dict[str, Any]:
    """READY when at least one model of every required kind passes all of its checks."""
    kinds = {"classification": [m for m in models.values() if m["kind"] == "classification"],
             "detection": [m for m in models.values() if m["kind"] == "detection"]}
    need = ["classification", "detection"] if require == "both" else [require]
    per_kind = {k: any(m["ready"] for m in kinds[k]) for k in need}
    return {"required": need, "per_kind_ready": per_kind, "ready": all(per_kind.values())}


# ------------------------------------------------------------------------------------------- stress tests
def stress_image(img: Image.Image, kind: str, seed: int = 0) -> Image.Image:
    """Apply one perturbation: gaussian_noise | blur | dark | bright | rotate."""
    img = img.convert("RGB")
    if kind == "gaussian_noise":
        rng = np.random.default_rng(seed)
        a = np.asarray(img).astype(np.float32) + rng.normal(0, 25, (img.height, img.width, 3))
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if kind == "blur":
        return img.filter(ImageFilter.GaussianBlur(2))
    if kind == "dark":
        return ImageEnhance.Brightness(img).enhance(0.5)
    if kind == "bright":
        return ImageEnhance.Brightness(img).enhance(1.6)
    if kind == "rotate":
        return img.rotate(25, resample=Image.BILINEAR, fillcolor=(0, 0, 0))
    raise ValueError(f"unknown stress '{kind}'")


STRESSES = ("gaussian_noise", "blur", "dark", "bright", "rotate")


def robustness_drop(clean: float, stressed: Dict[str, float]) -> float:
    """Worst relative macro-F1 drop over all stresses (0 if nothing got worse)."""
    if clean <= 0:
        return 1.0
    return float(max(0.0, max((clean - v) / clean for v in stressed.values())))


# ------------------------------------------------------------------------------------------- measurement
def process_memory_mb() -> Optional[float]:
    """Resident memory of this process (psutil if installed, else ``resource`` on Linux/macOS)."""
    try:
        import psutil
        return float(psutil.Process().memory_info().rss / 1024 ** 2)
    except ImportError:
        pass
    try:
        import resource
        v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return float(v / 1024 ** 2 if sys.platform == "darwin" else v / 1024)
    except Exception:
        return None


def _test_items(root: Path, names: Sequence[str], max_images: Optional[int], seed: int) -> Tuple[List[Path], np.ndarray]:
    paths, labels = [], []
    for i, c in enumerate(names):
        d = root / "test" / c
        if d.is_dir():
            for p in sorted(d.iterdir()):
                if p.suffix.lower() in IMG_EXT:
                    paths.append(p); labels.append(i)
    if not paths:
        raise FileNotFoundError(f"No test images under {root / 'test'}")
    labels = np.asarray(labels)
    if max_images and len(paths) > max_images:
        keep = np.random.default_rng(seed).permutation(len(paths))[:max_images]
        paths, labels = [paths[i] for i in keep], labels[keep]
    return paths, labels


def _predict(model, paths: Sequence[Path], size: int, device, transform_img: Callable[[Image.Image], Image.Image] = lambda x: x):
    import torch
    from torchvision import transforms as T
    from src.vision.common import IMAGENET_MEAN, IMAGENET_STD
    tf = T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    preds = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(paths), 32):
            x = torch.stack([tf(transform_img(Image.open(p).convert("RGB"))) for p in paths[i:i + 32]]).to(device)
            preds.append(model(x).argmax(1).cpu().numpy())
    return np.concatenate(preds)


def evaluate_classifier(name: str, cfg: Dict[str, Any], synthetic: bool, image_size: int, max_images: Optional[int]) -> Dict[str, Any]:
    """Load ``outputs/checkpoints/<name>.pt``, measure quality + speed + memory + robustness on the test split."""
    import torch
    from src.vision.classification import build_classifier
    from src.vision.common import checkpoint_path, classification_root, get_device
    ck_path = checkpoint_path(cfg, name)
    if not ck_path.exists():
        return {"kind": "classification", "model": name, "ready": False, "error": f"checkpoint not found: {ck_path}",
                "checks": check_metrics({}, cfg["deployment"], CLASSIFIER_CHECKS)}
    names, device = cfg["classes"]["names"], get_device()
    ck = torch.load(ck_path, map_location=device, weights_only=False)
    model = build_classifier(name, len(ck.get("classes", names))).to(device)
    model.load_state_dict(ck["state_dict"])
    paths, y = _test_items(classification_root(cfg, synthetic), names, max_images, cfg["project"]["seed"])
    pred_clean = _predict(model, paths, image_size, device)
    clean = classification_report(y, pred_clean, names)
    pur = recycling_purity(y, pred_clean, names, cfg["classes"]["recyclable"])
    stressed = {s: classification_report(y, _predict(model, paths, image_size, device, lambda im, s=s: stress_image(im, s)), names)["macro_f1"]
                for s in STRESSES}
    x1 = torch.randn(1, 3, image_size, image_size, device=device)
    with torch.no_grad():
        lat = measure_latency(lambda: model(x1), warmup=3, runs=20)
    mem = max(process_memory_mb() or 0.0, peak_gpu_memory_mb()) or None
    metrics = {"macro_f1": clean["macro_f1"], "purity_rate": pur["purity_rate"], "fps": lat["fps"], "latency_ms": lat["latency_ms_mean"],
               "memory_mb": mem, "robustness_drop": robustness_drop(clean["macro_f1"], stressed)}
    rows = check_metrics(metrics, cfg["deployment"], CLASSIFIER_CHECKS)
    return {"kind": "classification", "model": name, "device": str(device), "image_size": image_size, "n_test_images": len(paths),
            "metrics": metrics, "stress_macro_f1": stressed, "clean_accuracy": clean["accuracy"], "checks": rows,
            "ready": model_ready(rows), "source": "measured on this machine"}


def evaluate_detector(name: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Read ``outputs/logs/vision/detection_<name>.json`` written in Phase 2."""
    import json
    p = resolve_path(cfg["paths"]["logs"]) / "vision" / f"detection_{name}.json"
    if not p.exists():
        return {"kind": "detection", "model": name, "ready": False, "error": f"log not found: {p}",
                "checks": check_metrics({}, cfg["deployment"], DETECTOR_CHECKS)}
    d = json.loads(p.read_text())
    metrics = {"map50": d.get("map50"), "map50_95": d.get("map50_95"), "fps": d.get("fps"), "latency_ms": d.get("latency_ms_mean")}
    rows = check_metrics(metrics, cfg["deployment"], DETECTOR_CHECKS)
    return {"kind": "detection", "model": name, "metrics": metrics, "checks": rows, "ready": model_ready(rows),
            "source": f"Phase 2 log ({p.name}); speed measured where it was trained"}


def check_monitoring(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Informational: do the Phase 3 monitoring components load and run? (does not gate the verdict)"""
    import torch
    from src.advanced.lstm import forecast_next, load_lstm
    from src.advanced.vae import load_vae
    from src.vision.common import checkpoint_path
    out: Dict[str, Any] = {}
    p = checkpoint_path(cfg, "lstm_defect")
    if p.exists():
        try:
            m, meta = load_lstm(p)
            v = forecast_next(m, meta, np.full(meta["seq_len"], 0.1, dtype=np.float32))
            out["lstm_defect"] = PASS if np.isfinite(v) else FAIL
        except Exception as exc:
            out["lstm_defect"] = f"{FAIL}: {exc}"
    else:
        out["lstm_defect"] = NOT_EVAL
    p = checkpoint_path(cfg, "vae_main")
    if p.exists():
        try:
            m, meta = load_vae(p)
            with torch.no_grad():
                ok = m(torch.rand(1, 3, meta["image_size"], meta["image_size"]))[0].shape[-1] == meta["image_size"]
            out["vae_main"] = PASS if ok else FAIL
        except Exception as exc:
            out["vae_main"] = f"{FAIL}: {exc}"
    else:
        out["vae_main"] = NOT_EVAL
    return out


# ------------------------------------------------------------------------------------------- report
def to_markdown(report: Dict[str, Any]) -> str:
    v = report["verdict"]
    lines = [f"# Deployment readiness: {'READY' if v['ready'] else 'NOT READY'}", "",
             f"Required: {', '.join(v['required'])}  |  per kind: {v['per_kind_ready']}", ""]
    for key, m in report["models"].items():
        lines += [f"## {key} ({m['kind']}) - {'ready' if m['ready'] else 'not ready'}", ""]
        if m.get("error"):
            lines += [f"> {m['error']}", ""]
        lines += ["| metric | value | rule | status |", "|---|---|---|---|"]
        for r in m["checks"]:
            val = "n/a" if r["value"] is None else f"{r['value']:.4g}"
            lines.append(f"| {r['metric']} | {val} | {r['op']} {r['threshold']} | {r['status']} |")
        lines.append("")
    lines += ["## Monitoring components (informational)", ""] + [f"- {k}: {s}" for k, s in report["monitoring"].items()]
    return "\n".join(lines) + "\n"


def run_check(cfg: Optional[Dict[str, Any]] = None, synthetic: bool = False, image_size: int = 224, max_images: Optional[int] = 300,
              require: str = "both") -> Dict[str, Any]:
    cfg = cfg or load_config()
    set_seed(cfg["project"]["seed"])
    models: Dict[str, Dict[str, Any]] = {}
    for n in cfg["vision"]["classification"]["models"]:
        logger.info("classifier %s ...", n)
        models[n] = evaluate_classifier(n, cfg, synthetic, image_size, max_images)
    for n in ("fasterrcnn", "yolov8"):
        models[n] = evaluate_detector(n, cfg)
    report = {"thresholds": cfg["deployment"], "models": models, "verdict": overall_verdict(models, require),
              "monitoring": check_monitoring(cfg), "synthetic": synthetic}
    save_json(report, "deployment_report", "deployment", cfg)
    (ensure_dir(Path(cfg["paths"]["logs"]) / "deployment") / "deployment_report.md").write_text(to_markdown(report), encoding="utf-8")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="Deployment readiness check")
    ap.add_argument("--synthetic", action="store_true", help="synthetic test crops (smoke test only)")
    ap.add_argument("--image-size", type=int, default=224, help="must match the size the classifiers were trained with")
    ap.add_argument("--max-images", type=int, default=300, help="cap on test images for the stress tests (0 = all)")
    ap.add_argument("--require", choices=["both", "classification", "detection"], default="both")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    rep = run_check(load_config(a.config), a.synthetic, a.image_size, a.max_images or None, a.require)
    print((resolve_path("outputs/logs/deployment") / "deployment_report.md").read_text(encoding="utf-8"))
    return 0 if rep["verdict"]["ready"] else 1


if __name__ == "__main__":
    sys.exit(main())
