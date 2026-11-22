# Project 07 - Waste Sorting & Recycling Quality Controller (CSAI3027P)

Phase 1 implemented: `src/data`, `src/math_sandbox`, `src/fundamentals`, `configs/config.yaml`.
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
