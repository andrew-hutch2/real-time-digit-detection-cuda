# Camera-Based Digit Recognition Retraining System

This folder contains all the Python scripts for collecting training data from your camera and retraining the neural network model to improve accuracy for your specific preprocessing pipeline. This system integrates with the optimized camera recognition pipeline (`detectDigit_fast.py`) to collect high-quality training data.

## Scripts Overview

- **`retraining_workflow.py`** - Main workflow orchestrator that runs the complete retraining process
- **`data_collection.py`** - Captures digit samples from camera using your existing detection pipeline
- **`labeling_interface.py`** - Interactive tool for manually labeling collected samples
- **`data_validation.py`** - Validates quality and consistency of collected data
- **`data_organizer.py`** - Organizes labeled data into train/validation/test splits

## Overview

The retraining workflow consists of 5 main steps:

1. **Data Collection** - Capture digit samples from your camera
2. **Data Validation** - Check quality and consistency of collected data
3. **Data Labeling** - Manually label the collected samples
4. **Data Organization** - Split data into training/validation/test sets
5. **Model Retraining** - Fine-tune the model with your collected data

## Quick Start

### Complete Workflow (Recommended)
```bash
# Run the complete retraining workflow
make retrain-workflow
```

This will guide you through all steps automatically.

### Individual Steps
```bash
# Step 1: Collect data from camera
make collect-data

# Step 2: Validate collected data
make validate-data

# Step 3: Label the data (interactive)
make label-data

# Step 4: Organize data into splits
make organize-data

# Step 5: Retrain the model
make retrain-model
```

## Directory Structure

The scripts expect the following directory structure:

```
camera/
├── retraining/           # This folder (all Python scripts)
├── collected_data/       # Raw collected samples (created by data_collection.py)
│   ├── raw_samples/      # Original digit images
│   ├── preprocessed/     # Preprocessed binary data
│   ├── metadata/         # Sample metadata and labels
│   └── validation_report.json
└── training_data/        # Organized training data (created by data_organizer.py)
    ├── train/            # Training samples (70%)
    │   ├── 0/            # Digit 0 samples
    │   ├── 1/            # Digit 1 samples
    │   └── ...
    ├── val/              # Validation samples (20%)
    ├── test/             # Test samples (10%)
    └── training_manifest.json
```

## Detailed Instructions

### Step 1: Data Collection

The data collection script captures digit samples from your camera stream and preprocesses them using the same pipeline as your inference system.

```bash
# Basic data collection (200 samples)
make collect-data

# Or run directly with custom parameters
cd camera/retraining
python3 data_collection.py --target-samples 500 --output-dir ../my_data
```

**Controls during collection:**
- `q` - Quit collection
- `s` - Save current batch
- `r` - Reset counters

**Tips:**
- Collect samples in good lighting conditions
- Vary the digit sizes and positions
- Include different writing styles if possible
- Aim for at least 20-50 samples per digit

### Step 2: Data Validation

The validation script checks the quality and consistency of your collected data.

```bash
make validate-data
```

This will:
- Check for missing or corrupted files
- Validate data format and ranges
- Generate quality metrics
- Create a validation report
- Show data distribution visualization

### Step 3: Data Labeling

The labeling interface allows you to manually label the collected digit samples.

```bash
make label-data
```

**Controls:**
- `0-9` - Label digit
- `n` - Next sample
- `p` - Previous sample
- `s` - Skip sample
- `q` - Quit and save
- `r` - Reset current sample

**Tips:**
- Be consistent with your labeling
- Skip unclear or ambiguous samples
- Take breaks to avoid fatigue
- Aim for balanced distribution across digits

### Step 4: Data Organization

The organization script splits your labeled data into training, validation, and test sets.

```bash
make organize-data
```

Default split ratios:
- Training: 70%
- Validation: 20%
- Test: 10%

### Step 5: Model Retraining

The retraining script fine-tunes your existing model with the collected data.

```bash
make retrain-model
```

**Parameters:**
- Epochs: 50 (default)
- Learning rate: 0.001 (default)
- Early stopping: Enabled
- Best model: Automatically saved

## Advanced Usage

### Custom Parameters

```bash
# Run complete workflow with custom parameters
cd camera/retraining
python3 retraining_workflow.py --step all --target-samples 1000 --epochs 100 --learning-rate 0.0005
```

### Individual Script Usage

```bash
# Data collection with custom settings
python3 data_collection.py --target-samples 500 --camera-url http://your-camera-url

# Data validation with visualization
python3 data_validation.py --data-dir ../collected_data --visualize

# Data organization with custom splits
python3 data_organizer.py --train-ratio 0.8 --val-ratio 0.15 --test-ratio 0.05

# Direct retraining
../../bin/retrain ../training_data/ ../../bin/trained_model_weights.bin 100 0.0005
```

### Workflow Status

Check the current status of your retraining workflow:

```bash
make workflow-status
```

## Dependencies

- OpenCV (cv2)
- NumPy
- Matplotlib (for visualization)
- CUDA toolkit (for model retraining)

The scripts automatically handle relative paths and imports from the parent camera directory.

## Troubleshooting

### Common Issues

**1. No samples collected**
- Check camera connection
- Verify camera URL in data_collection.py
- Ensure good lighting conditions

**2. Poor detection quality**
- Adjust MSER parameters in detectDigit_fast.py
- Improve lighting conditions
- Use higher contrast backgrounds
- Use the reset functionality ('r' key) in camera.py

**3. Compilation errors**
- Ensure CUDA toolkit is installed
- Check nvcc is in PATH
- Verify all source files exist

**4. Insufficient training data**
- Collect more samples
- Use data augmentation
- Lower minimum samples requirement

### Performance Tips

**1. Data Collection**
- Use consistent lighting
- Collect diverse samples
- Include edge cases

**2. Labeling**
- Be consistent and accurate
- Skip ambiguous samples
- Take breaks to maintain quality

**3. Retraining**
- Start with lower learning rates
- Use early stopping
- Monitor validation accuracy

## Integration with Existing System

After retraining, update your inference system to use the new model:

```bash
# Copy retrained model to inference location
cp bin/retrained_model_best.bin bin/retrained_model_best250epoch.bin

# Test with new model
python3 camera.py
```

## Monitoring and Evaluation

### Validation Metrics
- Accuracy on validation set
- Per-digit accuracy
- Confusion matrix
- Loss curves

### Testing
```bash
# Test retrained model
make test-camera

# Compare with original model
python3 camera/evaluate_model.py --model retrained_model_best.bin
```

## Best Practices

1. **Data Quality**
   - Collect diverse, high-quality samples
   - Maintain consistent preprocessing
   - Validate data before training

2. **Training**
   - Use appropriate learning rates
   - Monitor overfitting
   - Save checkpoints regularly

3. **Evaluation**
   - Test on unseen data
   - Compare with baseline
   - Monitor real-world performance

4. **Iteration**
   - Collect more data if needed
   - Adjust preprocessing if necessary
   - Retrain with better parameters

## Support

For issues or questions:
1. Check the validation report for data quality issues
2. Review the workflow status
3. Examine error logs in the console output
4. Verify all dependencies are installed

## Example Workflow

Here's a complete example of retraining your model:

```bash
# 1. Collect 300 samples from camera
make collect-data
# (Follow on-screen instructions, press 'q' when done)

# 2. Validate the collected data
make validate-data

# 3. Label the samples
make label-data
# (Use 0-9 keys to label, 'n' for next, 'q' to quit)

# 4. Organize into train/val/test splits
make organize-data

# 5. Retrain the model
make retrain-model

# 6. Check workflow status
make workflow-status

# 7. Test the new model
python3 camera.py
```

Your retrained model will be saved as `bin/retrained_model_best.bin` and should show improved accuracy on your camera data! The camera system will automatically use the latest model weights.