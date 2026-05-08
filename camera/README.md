# Camera Pipeline (Real-Time Digit Detection and Recognition)

This directory contains the real-time webcam pipeline used by the project. It detects handwritten digits in video frames, preprocesses them into MNIST-style 28×28 inputs, and runs CUDA inference to display predictions.

## Overview

- **Detection**: MSER-based region proposals with filtering, duplicate removal, and multi-scale support.
- **Preprocessing**: transforms each detected ROI into a 28×28 “MNIST-like” input (polarity, centering, normalization).
- **Real-time performance**: non-blocking inference (threading) plus caching to avoid redundant predictions.
- **WSL2 support**: optional Windows MJPEG server for Linux development environments without direct webcam access.

## Quick start

### Prerequisites
- Python 3.8+
- OpenCV + NumPy (see `requirements.txt`)
- CUDA inference binary built in the repo root (`make inference` or `make all`)

### Run the camera app

```bash
python3 camera.py
```

To run detection-only (useful for tuning detection and preprocessing):

```bash
python3 detectDigit_fast.py
```

## WSL2 setup (webcam streaming)

WSL2 typically cannot access the host webcam directly. Use the included MJPEG server on Windows and point the client to that stream.

1) Start the webcam server on Windows:

```powershell
pip install flask opencv-python
python webcam_server.py
```

2) Update the stream URL used by the client (see `camera.py`).

3) Run the client:

```bash
python3 camera.py
```

## Controls

- `q`: quit
- `f`: toggle fullscreen
- `r`: reset detection state (useful after repositioning paper)

## How it’s structured

- `detectDigit_fast.py`: MSER detection + ROI extraction + MNIST-style preprocessing
- `camera.py`: real-time loop, threading, caching, overlay rendering
- `webcam_server.py`: optional Windows MJPEG streaming server (WSL2 workflow)

For retraining/fine-tuning the model using camera-collected samples, see `retraining/README.md`.