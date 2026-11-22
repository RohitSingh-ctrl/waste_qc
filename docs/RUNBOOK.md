# Runbook (what to run where)

Split: **VS Code** = code, tests, data pipeline, small checks. **Colab/Kaggle GPU** = heavy training. **Drive** = datasets and saved outputs.
**GitHub** = source only (`data/` and `outputs/` are git-ignored).

## 1. VS Code (local)
```bash
python -m venv .venv && .venv\Scripts\activate     # Windows
pip install -r requirements.txt
python -m pytest -q tests                           # all phases
python -m src.data.pipeline --source trashnet --download   # once
python -m src.data.pipeline --source taco --download       # once
```

## 2. Phase 3 on the laptop (CPU is enough)
```bash
python -m src.advanced.lstm_monitor                      # simulated stream, class mix from real train split
python -m src.advanced.vae --task all --image-size 32    # compare | latent2d | generate
python -m src.advanced.vae --task compare --epochs 5 --per-class 100   # faster CPU run
```
Add `--synthetic` for a no-data smoke test.

## 3. GPU training (Colab / Kaggle)
```python
!python -m src.vision.classification --model all
!python -m src.vision.unet
!python -m src.vision.detection --model all
!python -m src.experiments.ablations
```
Copy `outputs/` back to Drive after every run.

## 4. After training (any machine that has the checkpoints and JSON files)
```bash
python -m src.evaluation.robustness --model all --image-size 224   # must match the training image size
python -m src.evaluation.deployment                                # writes outputs/deployment_report.md
streamlit run app/gui.py
```

## 5. GitHub
```bash
git add . && git commit -m "Phase 3" && git push
```
