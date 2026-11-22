# Model card - Phase 3 models

## LSTM defect-trend monitor
* **Task:** forecast the next batch's defect (contamination) rate; alert when forecast >= 0.35.
* **Data:** simulated stream by default (labelled `source: simulated` in the JSON and in the GUI) or a user-supplied CSV.
* **Evaluation:** chronological 70 / 15 / 15 split, MAE / RMSE versus persistence and moving average, alert precision /
  recall / F1, event-level catch rate and lead time. See `outputs/logs/advanced/lstm_defect.json`.
* **Limits:** results on simulated data show the pipeline works, not how a real line behaves. The test segment holds only
  a handful of spike events, so alert metrics are noisy. Retrain on real logs.

## Convolutional VAE
* **Task:** unsupervised representation, reconstruction, novelty score, rare-class sampling.
* **Data:** TrashNet processed splits resized to 64 x 64 (config `advanced.vae.image_size`).
* **Evaluation:** test per-pixel MSE / PSNR / KL, activation sparsity and dead channels, latent plots.
  See `outputs/logs/advanced/vae_summary.json` and `outputs/figures/advanced/`.
* **Limits:** VAEs give blurry reconstructions. Generated `trash` images are only for augmentation experiments: never
  add them to validation or test, and measure their effect on classifier macro-F1 before using them. The reconstruction
  error is a weak novelty signal (simple but unusual images can reconstruct well).

## Results to report (fill from the JSON / figures after the real run)
| Item | Value | Source |
|---|---|---|
| LSTM MAE vs persistence | | `lstm_defect.json` |
| LSTM alert F1 | | `lstm_defect.json` |
| Best VAE activation (val loss) | | `vae_summary.json` |
| VAE test MSE / PSNR (z = 16) | | `vae_summary.json` |
| Deployment verdict | | `deployment_report.md` |
