# Technical report - fill-in template

Fill every value from your own `outputs/logs/**.json` files. Do not copy numbers from smoke tests (`--synthetic`, 1 epoch).

## 1. Classification (TrashNet)
| Model | Epochs | Pretrained | Accuracy | Macro-F1 | Purity | Contamination | FPS | Latency ms | Size MB |
|---|---|---|---|---|---|---|---|---|---|
| ResNet-50 | | | | | | | | | |
| GoogLeNet | | | | | | | | | |

## 2. Segmentation and detection (TACO)
| Model | mIoU / mAP@0.5 | mAP@0.5:0.95 | FPS | Size MB |
|---|---|---|---|---|
| U-Net | | - | | |
| Faster R-CNN | | | | |
| YOLOv8 | | | | |

## 3. Ablations (`outputs/logs/experiments/ablations.json`)
One factor at a time against the baseline; report test accuracy and what changed.

## 4. Advanced (Phase 3)
- LSTM monitor (simulated stream): MAE and spike precision/recall for LSTM vs persistence vs moving average (`outputs/logs/advanced/lstm_monitor.json`).
- VAE: validation pixel-MSE for step / sigmoid / tanh, 2-D latent plot, generated rare-class samples.

## 5. Robustness and deployment
- Worst stress and relative F1 drop per classifier (`robustness_*.json`).
- System verdict and best READY model per task (`outputs/deployment_report.md`).

## 6. Limitations
Copy the list from `docs/ARCHITECTURE.md` and add anything specific to your runs.
