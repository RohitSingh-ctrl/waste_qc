# Architecture

## Data flow
```
TrashNet / TACO (raw)
   -> src/data/pipeline.py: download, near-duplicate removal (image hash), group-aware split (no leakage),
      COCO -> YOLO / masks / crops
   -> data/processed/{trashnet,taco}/...
        |-- classification folders  -> src/vision/classification.py (ResNet-50, GoogLeNet)
        |-- masks                   -> src/vision/unet.py (U-Net from scratch)
        |-- YOLO folders            -> src/vision/detection.py (Faster R-CNN, YOLOv8)
   -> outputs/logs/vision/*.json  (accuracy, macro-F1, purity, mAP, FPS, latency, size)
   -> src/evaluation/robustness.py (stress tests)  -> outputs/logs/vision/robustness_*.json
   -> src/evaluation/deployment.py -> outputs/deployment_report.md
   -> app/gui.py (Streamlit) reads checkpoints + results
```

## Modules
| Folder | Purpose |
|---|---|
| `src/data` | Real-data pipeline, dedup, leakage-free split, converters, opt-in synthetic data (tests only) |
| `src/math_sandbox` | Tensor basics, SVD compression, gradient checking |
| `src/fundamentals` | McCulloch-Pitts / sigmoid neurons, NumPy MLP, Batch GD / SGD / Momentum / Nesterov |
| `src/experiments` | Ablations (init, batch norm, weight decay, dropout, augmentation), learning rules, conv visualisation |
| `src/vision` | ResNet-50, GoogLeNet, U-Net, Faster R-CNN, YOLOv8 |
| `src/advanced` | LSTM defect-trend monitor, Convolutional VAE (activation comparison, 2-D latent, rare-class generation) |
| `src/evaluation` | Metrics, benchmarking, robustness stress tests, deployment readiness check |
| `app` | `inference.py` (testable helpers) and `gui.py` (Streamlit) |
| `tests` | pytest suites for each phase |

## Design decisions
- **Real data only for real runs.** Synthetic data is opt-in (`--synthetic`) and only for smoke tests.
- **No leakage.** Splits are grouped by scene; the LSTM split is chronological with a purge gap of `seq_len` windows.
- **Recycling purity** is the headline industry metric: of the items sent to the recyclable stream, how many truly are recyclable.
- **Every model run writes** a JSON result, a checkpoint and figures, so the deployment check can read them without re-running anything.

## Known limitations (state these in the report)
- The LSTM monitor is evaluated on a **simulated** stream (class mix from real data, injected contamination episodes). It shows the
  pipeline works and how it compares with baselines; it is not field performance.
- The step activation in the VAE uses a straight-through estimator (a gradient approximation).
- The VAE for the rare class is trained on very few images; samples will be blurry and are an augmentation *candidate*, not a validated one.
- Classification checkpoints do not store the image size used in training; pass the same `--image-size` to robustness and the GUI.
- FPS / latency depend on the hardware that measured them (Colab GPU vs laptop CPU).
- Pretrained weights are off by default (`vision.classification.pretrained: false`), so scores are from-scratch scores.
- Mask R-CNN is not implemented; segmentation is U-Net only.
- Deployment memory criterion uses model size in MB as a proxy; peak runtime memory is not measured.
