# Deployment

## Readiness check
```bash
python -m src.deployment.check --image-size 224
```
Writes `outputs/logs/deployment/deployment_report.{json,md}`; exit code 0 = READY, 1 = NOT READY.

| Check | Rule (config `deployment:`) | Applies to | Source |
|---|---|---|---|
| macro-F1 | >= 0.85 | classifiers | measured on test split |
| purity rate | >= 0.95 | classifiers | measured |
| FPS / latency | >= 15 FPS, <= 80 ms | classifiers, detectors | classifiers measured here, detectors from Phase 2 log |
| memory | <= 2048 MB | classifiers | process RSS / peak GPU memory |
| robustness | <= 15 % relative macro-F1 drop | classifiers | noise, blur, dark, bright, rotate |
| mAP@0.5 / mAP@0.5:0.95 | >= 0.70 / >= 0.45 | detectors | Phase 2 log |

READY needs at least one classifier and one detector to pass every check (`--require` changes that). A missing
checkpoint, log or measurement is `NOT_EVALUATED` and counts as not ready. The monitoring components (LSTM, VAE) are
reported for information and do not gate the verdict.

## Run the GUI
```bash
streamlit run app/gui.py --server.port 8501
```
For a shared machine add `--server.address 0.0.0.0` and put it behind your normal access control; the GUI has no
authentication of its own.

## Production notes
* Measure speed on the target hardware: the check re-benchmarks classifiers on the machine it runs on, but detector speed
  comes from where Phase 2 trained it.
* Keep the config thresholds under version control; change them only with the line owner's sign-off.
* Re-run the check after every retrain. Retrain the LSTM on real defect logs and re-tune `spike_threshold`.
