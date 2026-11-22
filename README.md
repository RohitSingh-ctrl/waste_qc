# Project 07 - Waste Sorting & Recycling Quality Controller (CSAI3027P)

Phase 1: `src/data`, `src/math_sandbox`, `src/fundamentals`. Phase 2: `src/experiments`, `src/vision`, `src/evaluation`, `tests/`.
**Real data required:** TrashNet (and TACO for detection/segmentation) must be downloaded; missing data raises a clear error and is never silently replaced. Synthetic data is opt-in for unit tests only. Figures go to `outputs/figures/`, logs to `outputs/logs/`.

```bash
pip install -r requirements.txt
python -m src.data.pipeline --source trashnet --download   # dedup -> grouped split -> folders
python -m src.data.pipeline --source taco --download       # + YOLO/COCO/masks/crops
python -m src.math_sandbox.tensor_basics
python -m src.math_sandbox.svd_compression
python -m src.math_sandbox.gradient_checker
python -m src.fundamentals.neurons
python -m src.fundamentals.compare_optimizers
```
If the auto-download fails, put TrashNet class folders in `data/raw/trashnet/` (or TACO `annotations.json` + images in `data/raw/taco/`) and re-run without `--download`.

## Phase 2 (after the data pipeline)
```bash
python -m pytest -q tests
python -m src.experiments.learning_rules
python -m src.experiments.conv_visualization
python -m src.experiments.ablations                      # add --synthetic for a smoke test
python -m src.vision.classification --model all          # ResNet-50 + GoogLeNet (GPU recommended)
python -m src.vision.unet
python -m src.vision.detection --model all               # Faster R-CNN + YOLOv8
```
Add `--synthetic --epochs 1 --image-size 64` to any vision command for a quick CPU smoke test.
