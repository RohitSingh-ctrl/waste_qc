"""Minimal Streamlit GUI: classify uploaded items, see the sort decision, browse results and the deployment report.

Run:  streamlit run app/gui.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from PIL import Image  # noqa: E402

from app.inference import batch_summary, list_classifiers, load_classifier, predict_image, route_item  # noqa: E402
from src.utils import load_config, resolve_path  # noqa: E402

cfg = load_config()
st.set_page_config(page_title="Waste QC", layout="wide")
st.title("Waste Sorting & Recycling Quality Controller")
tab_cls, tab_res, tab_dep = st.tabs(["Classify", "Results", "Deployment"])


@st.cache_resource(show_spinner=False)
def _model(name: str):
    return load_classifier(name, cfg)


with tab_cls:
    models = list_classifiers(cfg)
    if not models:
        st.info("No trained classifier found in the checkpoints folder. Train one first: "
                "`python -m src.vision.classification --model resnet50` (Colab GPU recommended).")
    else:
        c1, c2, c3 = st.columns(3)
        name = c1.selectbox("Model", models)
        size = c2.number_input("Image size used in training", 32, 512, 224, 32)
        min_conf = c3.slider("Manual-check below confidence", 0.0, 1.0, 0.5, 0.05)
        files = st.file_uploader("Upload item images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        if files:
            model, names = _model(name)
            decisions = []
            for f in files:
                img = Image.open(f)
                probs = predict_image(model, img, names, int(size))
                d = route_item(probs, cfg, min_conf)
                decisions.append(d)
                col_a, col_b = st.columns([1, 2])
                col_a.image(img, caption=f.name, width=180)
                col_b.markdown(f"**{d['label']}** ({d['confidence']:.0%}) -> **{d['action']}**")
                col_b.bar_chart(pd.Series(probs, name="probability"))
            s = batch_summary(decisions)
            st.subheader("Batch summary (predicted, not ground truth)")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Items", s["n"]); m2.metric("Recycle", s["recycle"])
            m3.metric("Reject", s["reject"]); m4.metric("Manual check", s["manual_check"])
            st.bar_chart(pd.Series(s["class_counts"], name="items"))

with tab_res:
    rows = []
    for sub in ("vision", "advanced", "experiments"):
        for d in {resolve_path(cfg["paths"]["logs"]) / sub, resolve_path("outputs/logs") / sub}:
            for p in sorted(d.glob("*.json")) if d.is_dir() else []:
                try:
                    r = json.loads(p.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                t = r.get("test") if isinstance(r.get("test"), dict) else {}
                rows.append({"file": f"{sub}/{p.name}", "accuracy": t.get("accuracy"), "macro_f1": t.get("macro_f1"),
                             "purity": t.get("purity_rate"), "map50": r.get("map50"), "fps": r.get("fps")})
    if rows:
        st.dataframe(pd.DataFrame(rows))
    else:
        st.info("No result files yet. Run the training commands first.")
    figs = sorted({p for d in {resolve_path(cfg["paths"]["figures"]), resolve_path("outputs/figures")} if d.is_dir()
                   for p in d.rglob("*.png")})
    for p in figs:
        st.image(str(p), caption=p.name)

with tab_dep:
    report = resolve_path(cfg["paths"]["outputs"]) / "deployment_report.md"
    if st.button("Run deployment check"):
        from src.evaluation.deployment import run_deployment_check
        run_deployment_check(cfg)
    if report.exists():
        st.markdown(report.read_text(encoding="utf-8"))
    else:
        st.info("No deployment report yet. Click the button above after training.")
