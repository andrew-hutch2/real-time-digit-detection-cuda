# Testing Directory

This directory contains organized testing tools for digit detection and preprocessing analysis.

## 📁 Directory Structure

```
testing/
├── mser_detection/          # MSER-based detection algorithms
├── realtime_detection/      # Real-time detection implementations  
├── preprocessing/           # Preprocessing analysis tools
├── legacy/                  # Legacy/experimental scripts
└── README.md               # This file
```

## 🎯 MSER Detection System

### Why MSER (Maximally Stable Extremal Regions)?
After testing multiple contour-based approaches, we chose MSER because it:
- ✅ **Finds stable regions** that remain consistent across different thresholds
- ✅ **Resists noise** from shadows, paper edges, and background variations  
- ✅ **Handles borders intelligently** by devaluing regions near screen edges
- ✅ **Works consistently** under varying lighting conditions
- ✅ **Eliminates false positives** from irrelevant edges

### Key Features
- **Border Weight Mask** - Prioritizes center regions over edge regions
- **Duplicate Removal** - Eliminates overlapping detections of the same digit
- **Smart Filtering** - Combines geometric constraints with border weighting
- **Step-by-step Visualization** - Shows each processing stage for debugging

## 📂 Scripts by Category

### 🔍 MSER Detection (`mser_detection/`)

#### 1. mser_preprocessing_visualizer.py ⭐ **RECOMMENDED**

**MSER-based digit detection with step-by-step visualization - the current best approach.**

**Usage:**
```bash
cd mser_detection
python3 mser_preprocessing_visualizer.py [stream_url]
```

**Controls:**
- `s` - Save current frame and all MSER preprocessing stages
- `q` - Quit
- `r` - Reset counter

**Key Features:**
- **MSER Detection** - Uses Maximally Stable Extremal Regions for robust digit detection
- **Border Weighting** - Automatically devalues regions near screen edges
- **Duplicate Removal** - Eliminates overlapping detections of the same digit
- **Step-by-step Visualization** - Shows each processing stage for debugging

**Output Files:**
1. `mser_XXX_01_original.jpg` - Original frame
2. `mser_XXX_02_grayscale.jpg` - Grayscale conversion  
3. `mser_XXX_03_blurred.jpg` - Gaussian blur (MSER preprocessing)
4. `mser_XXX_04_border_weight_mask.jpg` - Border weighting visualization
5. `mser_XXX_05_mser_regions.jpg` - All detected MSER regions
6. `mser_XXX_06_filtered_regions.jpg` - Regions that passed filtering
7. `mser_XXX_07_digit_X_roi.jpg` - Individual digit regions
8. `mser_XXX_08_digit_X_preprocessed.jpg` - 28x28 preprocessed digits
9. `mser_XXX_09_digit_X_enhanced.jpg` - Contrast-enhanced digits
10. `mser_XXX_10_detection_overlay.jpg` - Final detection overlay

#### 2. mser_digit_detector.py

**Core MSER-based digit detector class used by the visualizer.**

**Usage:**
```bash
cd mser_detection
python3 mser_digit_detector.py <image_file>
```

**Features:**
- **MSER Region Detection** - Finds stable regions across multiple thresholds
- **Border Weight Mask** - Prioritizes center regions over edge regions
- **Duplicate Removal** - Eliminates overlapping detections
- **Smart Filtering** - Combines geometric and weight-based filtering

#### 3. mser_digit_detection_accurate.py & mser_digit_detection_fast.py

**Standalone MSER detection implementations with different speed/accuracy tradeoffs.**

### ⚡ Real-time Detection (`realtime_detection/`)

Contains threaded real-time detection implementations:
- `realtime_mser_threaded.py` - Basic threaded MSER detection
- `realtime_mser_fast_threaded.py` - Optimized for speed
- `realtime_mser_fast_threaded2.py` - Further optimizations
- `realtime_mser_fast_threaded2.5.py` - Balanced approach
- `realtime_mser_fast_threaded3.py` - Latest optimizations
- `realtime_mser_fast_threaded4.py` - Most recent version
- `realtime_mser_accurate_threaded.py` - Prioritizes accuracy over speed

### 🔬 Preprocessing Analysis (`preprocessing/`)

#### preprocessing_visualizer.py ⭐ **NEW - INFERENCE DEBUGGING**

**Captures and visualizes preprocessed digits before inference to debug preprocessing issues.**

**Usage:**
```bash
cd preprocessing
python preprocessing_visualizer.py
```

**Controls:**
- **'c'** - Capture current detections (saves 5-10 digits per run)
- **'q'** - Quit the application
- **'s'** - Show preprocessing statistics

**Purpose:**
This tool helps debug why the model might be guessing only certain digits (like 3 or 8) by allowing you to:

- See exactly what the model receives as input
- Compare original camera input vs preprocessed data
- Analyze preprocessing statistics (mean, std, min, max values)
- Identify preprocessing issues that might cause poor classification

**Output for each captured digit:**
1. **Original ROI** - The raw digit region from the camera
2. **Preprocessed 28x28** - The digit after preprocessing (28x28 pixels)
3. **Enlarged 280x280** - The preprocessed digit enlarged 10x for visibility
4. **Comparison** - Side-by-side view of original vs preprocessed
5. **Data file** - Text file with preprocessing statistics

**Example Output Structure:**
```
preprocessing_output_20241201_143022/
├── digit_01_original.png
├── digit_01_preprocessed_28x28.png
├── digit_01_preprocessed_280x280.png
├── digit_01_comparison.png
├── digit_01_data.txt
├── digit_02_original.png
└── ...
```

### 🗂️ Legacy (`legacy/`)

Contains experimental and legacy scripts for reference and comparison.

## 🎯 Quick Start

### For MSER Detection (Best Results):
```bash
cd mser_detection
python3 mser_preprocessing_visualizer.py
```

### For Preprocessing Debugging:
```bash
cd preprocessing
python preprocessing_visualizer.py
```

### For Real-time Performance:
```bash
cd realtime_detection
python3 realtime_mser_fast_threaded4.py
```

## ⚙️ Configuration

### MSER Parameters (in `mser_digit_detector.py`)
```python
self.mser = cv2.MSER_create(
    delta=4,           # Delta parameter for stability
    min_area=500,      # Minimum area (reduced for more detections)
    max_area=80000,    # Maximum area (increased for larger digits)
    max_variation=0.3, # Maximum variation for stability
    min_diversity=0.15, # Minimum diversity between regions
)
```

### Border Weighting Parameters
```python
self.border_margin = 0.1  # 10% margin from edges
self.center_weight = 1.0
self.border_weight = 0.3  # Much lower weight for border regions
```

## 🎛️ Tuning Tips

### For Better Detection
1. **Adjust MSER parameters** if detecting too many/few regions
2. **Modify border margin** if screen edges are still problematic
3. **Tune size filters** based on your digit sizes
4. **Adjust weight threshold** (currently 0.2) for filtering

### For Different Resolutions
- **1080p**: Current settings should work well
- **720p**: Reduce `min_area` and `max_area` by ~50%
- **4K**: Increase `min_area` and `max_area` by ~200%

## 🐛 Troubleshooting

### Too Many False Positives
- Increase `min_area` and `max_area`
- Increase weight threshold (currently 0.2)
- Adjust `border_margin` to be larger

### Missing Digits
- Decrease `min_area` and `max_area`
- Decrease weight threshold
- Check if digits are too close to borders

### Model Only Guessing Certain Digits (3, 8, etc.)
- Use `preprocessing/preprocessing_visualizer.py` to capture and analyze preprocessed digits
- Check if digits look similar to MNIST training data
- Verify preprocessing normalization and contrast
- Ensure digits are properly centered and sized