# User guide

## 1. Install
```bash
pip install -r requirements.txt
```

## 2. Data and Phase 1-2 models
```bash
python -m src.data.pipeline --source trashnet --download
python -m src.data.pipeline --source taco --download
python -m src.vision.classification --model all       # GPU recommended
python -m src.vision.unet
python -m src.vision.detection --model all
```

## 3. Phase 3
```bash
python -m src.advanced.lstm [--csv my_log.csv] [--epochs N]
python -m src.advanced.vae  [--epochs N] [--activations relu silu] [--save-augmented]
python -m src.deployment.check [--image-size 224] [--max-images 300] [--require both|classification|detection]
streamlit run app/gui.py
```
`--synthetic` (VAE, deployment check) uses generated data for smoke tests only; numbers from it mean nothing.

## 4. Using the GUI
| Tab | What to do | Needs |
|---|---|---|
| Classify item | Pick a model and the input size used in training, upload an image | `outputs/checkpoints/resnet50.pt` or `googlenet.pt` |
| Defect trend | View the saved test forecast, upload a CSV, or use a simulated sample | `lstm_defect.pt` |
| Image inspector | Upload an image; large reconstruction error = unusual item | `vae_main.pt` |
| Deployment report | Read the last report or re-run the check | `deployment_report.json` (optional) |

A tab with a missing file shows the command that creates it.

## 5. Defect-log CSV format
```
defect_rate
0.12
0.10
```
or two columns `defects,items`. At least `seq_len` (30) rows are needed.

## 6. Output locations
| What | Where |
|---|---|
| Figures | `outputs/figures/` (`advanced/` for Phase 3) |
| Metrics / reports | `outputs/logs/` |
| Checkpoints | `outputs/checkpoints/` |

## 7. Troubleshooting
* `DataNotFoundError`: run the data pipeline first (or place the dataset in `data/raw/`).
* GUI classification output looks random: set the input size to the one used in training.
* `NOT_EVALUATED` in the report: that model's checkpoint / log is missing, so it cannot count as ready.
* Out of GPU memory in the VAE: lower `advanced.vae.batch_size` or `--image-size 32`.
