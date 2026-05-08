# Real-Time Handwritten Digit Recognition (CUDA + OpenCV)

## Results (summary)

Designed and deployed a real-time webcam digit recognition system, processing live streams at **30+ FPS** and achieving **90% real-world accuracy** after retraining. Implemented CUDA-accelerated inference for a custom **3-layer MLP** (cuBLAS + optimized GPU memory management) and engineered a multi-threaded pipeline with dedicated threads for capture, detection, and inference to maintain throughput. Built an end-to-end data collection and retraining workflow, capturing **500+** real-world digit samples via webcam and improving accuracy by **40%** over the original MNIST-trained baseline.

The CUDA MNIST implementation is based on Infatoshi’s [`mnist-cuda`](https://github.com/Infatoshi/mnist-cuda/tree/master) project (see `v5.cu`). This repo extends that baseline into a full real-time system and adds a workflow to adapt MNIST weights to real camera data.

## What’s in here

- **CUDA model**: a multi-layer perceptron (MLP) trained on MNIST-formatted binaries, with saved weights for reuse.
- **Real-time detection**: MSER-based digit detection in webcam frames, with filtering, duplicate removal, and multi-scale support.
- **MNIST-style preprocessing**: transforms detected regions into 28×28 “MNIST-like” inputs (centering/contrast/normalization) before inference.
- **Real-world adaptation (the “cool part”)**: collect real handwritten digits from your webcam, label them, and fine-tune the MNIST-trained model to significantly improve real-time predictions under your lighting/camera/writing style.

## Demo

- `https://github.com/user-attachments/assets/045e8b9a-2a98-4316-91e5-86caf4e21cd8`
- `https://github.com/user-attachments/assets/1f5abb35-ef3f-4ba7-a602-5fded76a2d3f`
- `https://github.com/user-attachments/assets/a5dbb7c6-4f08-472d-b004-9e2b540f48f3`

## Technical highlights

### CUDA training and inference
- **Architecture**: 784 → 1568 → 784 → 10 MLP.
- **Acceleration**: cuBLAS-backed matrix operations + CUDA kernels for activations.
- **Persistence**: binary weight saving/loading for inference and retraining workflows.

### Detection improvements (camera side)
- **MSER-based detection** for stable region proposals in messy webcam frames.
- **Filtering and duplicate removal** to reduce false positives and overlapping boxes.
- **Multi-scale processing** to handle different digit sizes and distances.
- **Non-blocking inference**: threaded pipeline and caching to keep UI responsive.

### MNIST-style preprocessing (why it matters)
The biggest gap between “MNIST works” and “webcam works” is input distribution. This project explicitly preprocesses camera detections to match MNIST conventions:

- Convert ROI to grayscale and suppress noise (blur).
- Threshold/invert so digits resemble MNIST polarity (bright digit on dark background).
- Resize and center to 28×28.
- Normalize using MNIST statistics (mean/std) prior to inference.

This makes the model see camera digits in a representation closer to its training data.

## Real-world adaptation: fine-tune MNIST on your digits
MNIST-trained weights are a strong starting point, but real camera data differs (lens, compression, lighting, paper texture, pen thickness, personal handwriting). This repo includes a workflow to:

1. **Collect** digit samples from the camera using the same detection + preprocessing pipeline used at inference time.
2. **Label** samples via a lightweight UI.
3. **Organize** into train/validation splits.
4. **Fine-tune** the MNIST-trained CUDA model on your collected samples with early stopping.

This step is designed to improve real-time accuracy on your exact setup.

## Quick start

### Prerequisites
- CUDA Toolkit (tested with 11+)
- Python 3.8+ (for the camera pipeline and retraining tools)

### Build CUDA binaries

```bash
make all
```

This builds:
- `bin/train2` (MNIST training)
- `bin/inference` (single-digit inference on a 28×28 float `.bin`)
- `bin/retrain` (fine-tuning on camera-collected samples)

### Run the real-time camera app
See `camera/README.md` for the recommended setup (including WSL2 streaming if needed):

```bash
cd camera
python3 camera.py
```

### Fine-tune on your own digits
See `camera/retraining/README.md` for the full workflow. From the repo root:

```bash
make retrain-workflow
```

## Repository layout

```
real-time-digit-detection-cuda/
├── README.md
├── Makefile
├── v5.cu                         # Upstream reference (mnist-cuda baseline)
├── cuda/                         # CUDA training/inference/retraining
├── bin/                          # Compiled executables and weights (generated)
└── camera/                       # Real-time detection + preprocessing + retraining tools
```

## Notes / known assumptions
- The MNIST training executable expects MNIST-formatted binary files under `../data/` (see `Makefile` / `cuda/train2.cu`). Those files are not included in this repo.
- For webcam access on WSL2, use the Windows MJPEG server (`camera/webcam_server.py`) and point the Linux client at the stream URL.

## Credits
- Upstream foundation: Infatoshi’s [`mnist-cuda`](https://github.com/Infatoshi/mnist-cuda/tree/master) (see `v5.cu`).
