# Camera Pipeline for MNIST Digit Recognition

This directory contains the image preprocessing pipeline for detecting and recognizing digits from webcam frames, with support for WSL2 webcam access via MJPEG streaming.

## Overview

The camera pipeline consists of several components:

1. **Digit Detection** (`digit_detector.py`) - Detects digit regions in images
2. **Camera Client** (`camera.py`) - Connects to MJPEG webcam stream from Windows
3. **Requirements** (`requirements.txt`) - Python dependencies

## WSL2 Webcam Setup

Since WSL2 doesn't have direct access to webcams, we use an MJPEG streaming solution:

### 1. Windows Server Setup

**Create `webcam_server.py` on Windows:**
copy and paste the webcam_server.py file
into your windows system and run it. This should create a simple flask
webserver that allows us to use the webcam feed in wsl2 with opencv

**Install dependencies and run:**
```powershell
pip install flask opencv-python
python webcam_server.py
```

### 2. WSL2 Client Setup

**Install dependencies:**
```bash
pip install opencv-python requests
```

**Update the IP address in `camera.py`** (line 12) to match your Windows IP:
```python
stream_url = "http://YOUR_WINDOWS_IP:8080/video"
```

**Run the camera client:**
```bash
python3 camera.py
```

## Usage

### Using with Webcam

```bash
# Start Windows server first, then run:
python3 camera.py
```

**Controls:**
- `q` - Quit
- `r` - Reset frame counter
- `s` - Save current frame

## How It Works

### 1. Digit Detection
- Converts frame to grayscale
- Applies Gaussian blur and adaptive thresholding
- Finds contours that match digit characteristics
- Filters by area, aspect ratio, and size

### 2. Digit Preprocessing
- Extracts digit regions with padding
- Resizes to 28x28 pixels
- Inverts colors (white digits on black background)
- Normalizes to [0,1] range
- Applies MNIST normalization (mean=0.1307, std=0.3081)

### 3. CUDA Inference
- Saves preprocessed digit to binary file
- Calls the CUDA inference program
- Parses prediction results
- Cleans up temporary files

## File Structure

```
camera/
├── digit_detector.py      # Core digit detection and preprocessing
├── camera.py             # MJPEG webcam client for WSL2
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── test_images/         # Test images for development
```

## Dependencies

- OpenCV (cv2) - Image processing and webcam capture
- NumPy - Numerical operations
- Flask - MJPEG streaming server (Windows)
- Requests - HTTP client for stream connection (WSL2)

## Integration with CUDA Model

The pipeline integrates with the trained CUDA model by:

1. Detecting digits in webcam frames
2. Preprocessing to match MNIST format
3. Saving to binary format
4. Calling `../bin/inference` program
5. Displaying results in real-time

## Troubleshooting

### Common Issues

1. **Connection refused**: Make sure Windows webcam server is running on port 8080
2. **Wrong IP address**: Update the IP in `camera.py` to match your Windows machine
3. **Firewall blocking**: Allow Python through Windows Firewall when prompted
4. **Camera not found**: Make sure webcam is connected and not used by other applications
5. **Inference executable not found**: Run `cd .. && make inference` first
6. **Poor digit detection**: Adjust detection parameters in `DigitDetector` class
7. **Low accuracy**: Ensure good lighting and clear digit contrast

### Performance Tips

- Use good lighting for better digit detection
- Write digits clearly and at reasonable size
- Avoid cluttered backgrounds
- Ensure digits have good contrast with background

## Future Enhancements

- Multi-digit number recognition
- Handwriting style adaptation
- Real-time performance optimization
- Better digit segmentation
- Confidence scoring
