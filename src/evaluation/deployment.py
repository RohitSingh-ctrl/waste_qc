"""Deployment readiness check: compare saved model results with the acceptance thresholds in ``configs/config.yaml``.

Reads the JSON files written by the earlier phases (``outputs/logs/vision/*.json``) and produces
``outputs/logs/deployment/deployment_check.json`` plus a human-readable ``outputs/deployment_report.md``.

Rules (kept deliberately strict and honest):
* a criterion with no measurement is ``not_measured`` (never silently passed);
* a model is READY only if every applicable criterion was measured and passed;
* the system verdict is READY only if there is a READY model for *each* task (classification and detection).

Latency / FPS depend on the machine that produced them (Colab GPU vs laptop CPU) - re-measure on the target hardware.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils import ensure_dir, get_logger, load_config, resolve_path, save_json

logger = get_logger(__name__)

# criterion -> (config key, direction, how to read the value from a result dict, applies to)
CRITERIA = [
    ("recycling purity", "min_purity_rate", ">=", lambda r: (r.get("test") or {}).get("purity_rate"), "classification"),
    ("macro-F1", "min_macro_f1", ">=", lambda r: (r.get("test") or {}).get("macro_f1"), "classification"),
    ("mAP@0.5", "min_map50", ">=", lambda r: r.get("map50"), "detection"),
    ("mAP@0.5:0.95", "min_map50_95", ">=", lambda r: r.get("map50_95"), "detection"),
    ("FPS", "min_fps", ">=", lambda r: r.get("fps"), "both"),
    ("latency (ms)", "max_latency_ms", "<=", lambda r: r.get("latency_ms_mean"), "both"),
    ("model size (MB, proxy for memory)", "max_memory_mb", "<=", lambda r: r.get("size_mb"), "both"),
    ("robustness drop", "max_robustness_drop", "<=", lambda r: (r.get("robustness") or {}).get("max_relative_drop"),
     "classification"),
]


def _load(p: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    except (OSError, json.JSONDecodeError):
        return None


def load_results(cfg: Dict[str, Any]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Collect saved results: ``{"classification": {model: result}, "detection": {...}}``."""
    # Phase 2 modules write to outputs/logs; a custom ``paths.logs`` is honoured too (first hit wins).
    dirs = [resolve_path(cfg["paths"]["logs"]) / "vision", resolve_path("outputs/logs") / "vision"]

    def find(name: str) -> Optional[Dict[str, Any]]:
        for d in dirs:
            r = _load(d / name)
            if r is not None:
                return r
        return None

    out: Dict[str, Dict[str, Dict[str, Any]]] = {"classification": {}, "detection": {}}
    for m in cfg["vision"]["classification"]["models"]:
        r = find(f"classification_{m}.json")
        if r is not None:
            r["robustness"] = find(f"robustness_{m}.json")
            out["classification"][m] = r
    for m in ("fasterrcnn", "yolov8"):
        r = find(f"detection_{m}.json")
        if r is not None:
            out["detection"][m] = r
    return out


def evaluate_model(result: Dict[str, Any], task: str, thresholds: Dict[str, float]) -> Dict[str, Any]:
    """Check one model against the thresholds that apply to its task."""
    rows: List[Dict[str, Any]] = []
    for label, key, op, getter, applies in CRITERIA:
        if applies not in (task, "both"):
            continue
        thr, val = thresholds[key], getter(result)
        if val is None:
            status = "not_measured"
        else:
            status = "pass" if (val >= thr if op == ">=" else val <= thr) else "fail"
        rows.append({"criterion": label, "threshold": f"{op} {thr}", "value": None if val is None else float(val),
                     "status": status})
    statuses = {r["status"] for r in rows}
    verdict = "FAIL" if "fail" in statuses else ("INCOMPLETE" if "not_measured" in statuses else "READY")
    return {"task": task, "verdict": verdict, "rows": rows}


def evaluate_deployment(results: Dict[str, Dict[str, Dict[str, Any]]], thresholds: Dict[str, float]) -> Dict[str, Any]:
    """Per-model verdicts, best candidate per task, and the overall system verdict."""
    models = {f"{t}/{m}": evaluate_model(r, t, thresholds) for t, group in results.items() for m, r in group.items()}
    best: Dict[str, Optional[str]] = {}
    for task in ("classification", "detection"):
        ready = [k for k, v in models.items() if v["task"] == task and v["verdict"] == "READY"]
        key = "macro_f1" if task == "classification" else "map50"
        if ready:
            def score(k: str) -> float:
                r = results[task][k.split("/", 1)[1]]
                return float((r.get("test") or {}).get(key, 0.0) if task == "classification" else r.get(key, 0.0))
            best[task] = max(ready, key=score)
        else:
            best[task] = None
    have = {v["task"] for v in models.values()}
    if not models:
        system = "INCOMPLETE"
    elif all(best[t] for t in ("classification", "detection")):
        system = "READY"
    elif any(v["verdict"] == "INCOMPLETE" for v in models.values()) or have != {"classification", "detection"}:
        system = "INCOMPLETE"
    else:
        system = "NOT READY"
    return {"system_verdict": system, "best_per_task": best, "models": models}


def render_report(check: Dict[str, Any]) -> str:
    icon = {"pass": "PASS", "fail": "FAIL", "not_measured": "not measured"}
    lines = ["# Deployment readiness report", "",
             f"**System verdict: {check['system_verdict']}**", "",
             f"Best READY classifier: {check['best_per_task']['classification'] or 'none'}  ",
             f"Best READY detector: {check['best_per_task']['detection'] or 'none'}", "",
             "A model is READY only if every applicable criterion was measured and passed. Latency/FPS depend on the "
             "hardware used for training/benchmarking; re-measure on the target device.", ""]
    for name, m in check["models"].items():
        lines += [f"## {name} - {m['verdict']}", "", "| criterion | threshold | value | status |", "|---|---|---|---|"]
        for r in m["rows"]:
            v = "-" if r["value"] is None else f"{r['value']:.4g}"
            lines.append(f"| {r['criterion']} | {r['threshold']} | {v} | {icon[r['status']]} |")
        lines.append("")
    if not check["models"]:
        lines.append("No result files found in `outputs/logs/vision/`. Run the Phase 2 training commands first.")
    return "\n".join(lines)


def run_deployment_check(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = cfg or load_config()
    check = evaluate_deployment(load_results(cfg), cfg["deployment"])
    save_json(check, "deployment_check", "deployment", cfg)
    report = resolve_path(cfg["paths"]["outputs"]) / "deployment_report.md"
    ensure_dir(report.parent)
    report.write_text(render_report(check), encoding="utf-8")
    check["report"] = str(report)
    logger.info("system verdict: %s | best per task: %s | report: %s", check["system_verdict"], check["best_per_task"], report)
    return check


def main() -> None:
    ap = argparse.ArgumentParser(description="Check saved model results against deployment thresholds")
    ap.add_argument("--config", default=None)
    run_deployment_check(load_config(ap.parse_args().config))


if __name__ == "__main__":
    main()
