# Retraining / Fine-Tuning on Camera Data

MNIST-trained weights are a strong baseline, but real webcam data differs (lighting, blur, paper texture, pen thickness, and personal handwriting). This folder contains tooling to **collect real digits**, **label them**, and **fine-tune** the CUDA model so real-time predictions improve on your specific setup.

## What this workflow does

1. **Collect** samples from the same camera detection + preprocessing pipeline used at inference time.
2. **Validate** that the collected samples are usable and consistent.
3. **Label** samples quickly with keyboard shortcuts.
4. **Organize** data into train/validation splits.
5. **Fine-tune** the CUDA model using the MNIST weights as initialization (transfer learning), with early stopping.

## Quick start (recommended)

From the repo root:

```bash
make retrain-workflow
```

## Individual steps

```bash
make collect-data
make validate-data
make label-data
make organize-data
make retrain-model
```

## Scripts

- `scripts/retraining_workflow.py`: orchestrates the full workflow
- `scripts/data_collection.py`: captures samples using the camera pipeline
- `scripts/labeling_interface.py`: interactive labeling UI
- `scripts/data_validation.py`: sanity checks and reporting
- `scripts/data_organizer.py`: split/organize for training

## Outputs

- Camera samples are stored as per-example `.bin` files containing **784 floats** (28×28 flattened) plus `.json` metadata.
- The CUDA retrainer saves the best checkpoint as `bin/retrained_model_best.bin` (see `cuda/retrain.cu`).

## Notes

- For best results, collect data under the same conditions you’ll run inference (same camera, distance, lighting, paper/background).
- If you’re using WSL2 streaming, ensure the collection step points at the same stream URL used by `camera.py`.