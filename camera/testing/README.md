# Testing Directory

This directory contains MSER-based digit detection scripts that solve common issues like paper edges and screen borders being detected as digits.

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

## Scripts

### 1. mser_preprocessing_visualizer.py ⭐ **RECOMMENDED**

**MSER-based digit detection with step-by-step visualization - the current best approach.**

**Usage:**
```bash
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

### 2. mser_digit_detector.py

**Core MSER-based digit detector class used by the visualizer.**

**Usage:**
```bash
python3 mser_digit_detector.py <image_file>
```

**Features:**
- **MSER Region Detection** - Finds stable regions across multiple thresholds
- **Border Weight Mask** - Prioritizes center regions over edge regions
- **Duplicate Removal** - Eliminates overlapping detections
- **Smart Filtering** - Combines geometric and weight-based filtering

### 3. preprocessing_visualizer.py

**Legacy contour-based detection (for comparison purposes only).**

**Usage:**
```bash
python3 preprocessing_visualizer.py [stream_url]
```

**Note:** This is the original contour-based approach that was replaced by MSER detection. Use `mser_preprocessing_visualizer.py` for best results.

## Purpose

These tools help you:
1. **Visually inspect** how preprocessing affects digit quality
2. **Identify issues** like poor thresholding or false positives from paper edges
3. **Compare MSER vs traditional** detection approaches
4. **Analyze region properties** to understand detection accuracy
5. **Evaluate improvements** in false positive reduction
6. **Debug detection issues** with step-by-step visualization
7. **Tune parameters** for optimal performance in your environment

## 🎯 Quick Start

**For best results, use the MSER-based detection:**
```bash
cd /home/ahutch/CUDA_test/mnist-mlp/digitsClassification/camera/testing
python3 mser_preprocessing_visualizer.py
```

**Key advantages of MSER approach:**
- ✅ **No more paper edge detection** - Border weighting eliminates false positives
- ✅ **Consistent performance** - Works reliably across different lighting conditions  
- ✅ **Cleaner results** - Fewer duplicate detections, better digit isolation
- ✅ **Visual debugging** - See exactly how border weighting affects detection

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
