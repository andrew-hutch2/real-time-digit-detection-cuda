# MNIST CUDA Neural Network with Real-Time Camera Recognition

This project is based on code from the GitHub repository [mnist-cuda](https://github.com/Infatoshi/mnist-cuda/tree/master), specifically the file `v5.cu` in this directory, which serves as the foundation for this enhanced implementation.

## 🎯 Project Overview

This project has two main objectives:

### 1. Neural Network Optimization
To optimize the accuracy and performance of the CUDA-based MNIST neural network to achieve above 95% testing accuracy with significant performance advancements through:

- **Enhanced Network Architecture**: Added two additional hidden layers to the MLP for more complex learning
- **Decaying Learning Rate**: Adaptive learning rate that slows down as accuracy improves for better fine-tuning
- **Dropout Regularization**: 20% dropout rate in ReLU layers to prevent overfitting and improve generalization
- **Weight Decay**: Added weight decay constant to discourage overfitting and encourage generalization
- **Model Persistence**: Implemented weight saving and loading functionality for model reuse

### 2. Real-Time Camera Recognition System
To create a system for recognizing real handwritten digits in real-time from OpenCV camera footage, solving the major challenge of:

- **Digit Detection**: MSER-based detection pipeline to find digits in webcam frames
- **Image Preprocessing**: Converting detected regions to 28x28 MNIST-compatible format
- **Real-Time Processing**: Optimized pipeline for live camera feed processing
- **Model Integration**: Seamless integration with the CUDA neural network for predictions

## 🚀 Key Features

### High-Performance CUDA Implementation
- **Multi-layer MLP**: 784 → 1568 → 784 → 10 architecture
- **CUDA Acceleration**: GPU-accelerated matrix operations using cuBLAS
- **Optimized Memory Management**: Efficient GPU memory allocation and management
- **Batch Processing**: Support for batch inference operations

### Advanced Camera Recognition Pipeline
- **MSER Detection**: Maximally Stable Extremal Regions for robust digit detection
- **Multi-Scale Processing**: Detection at multiple scales for better coverage
- **Border Weighting**: Intelligent edge detection with center-priority weighting
- **Non-Blocking Inference**: Asynchronous processing for smooth real-time performance
- **Smart Caching**: Hash-based caching to avoid redundant processing
- **Reset Functionality**: Manual and automatic reset for repositioned paper

### Comprehensive Retraining System
- **Data Collection**: Automated capture of digit samples from camera
- **Interactive Labeling**: User-friendly interface for manual data labeling
- **Data Validation**: Quality checks and consistency validation
- **Model Fine-tuning**: Retrain the model with camera-specific data
- **Performance Monitoring**: Real-time statistics and quality metrics

### Webcam-Specific Model Adaptation
The system includes a complete pipeline for adapting the pre-trained MNIST model to your specific webcam setup:

1. **Data Collection**: Capture digit samples using your actual camera and lighting conditions
2. **Quality Validation**: Automated checks for data consistency and quality
3. **Interactive Labeling**: Manual annotation of collected samples with keyboard shortcuts
4. **Data Organization**: Automatic splitting into training/validation/test sets
5. **Transfer Learning**: Fine-tune the pre-trained model with your camera data
6. **Performance Validation**: Test the adapted model on your specific setup

This process significantly improves accuracy by adapting the model to your specific:
- Camera characteristics and image quality
- Lighting conditions and environment
- Writing style and digit appearance
- Preprocessing pipeline variations

## 📁 Project Structure

```
digitsClassification/
├── README.md                    # This file
├── Makefile                     # Build and run commands
├── v5.cu                        # Original CUDA implementation (reference)
├── bin/                         # Compiled binaries and model weights
│   ├── inference               # CUDA inference executable
│   ├── retrain                 # CUDA retraining executable
│   └── *.bin                   # Model weight files
├── cuda/                        # CUDA source code
│   ├── neuralNetwork.cuh       # Network architecture definitions
│   ├── inference.cu            # Inference implementation
│   └── retrain.cu              # Training implementation
└── camera/                      # Camera recognition system
    ├── README.md               # Camera system documentation
    ├── camera.py               # Main camera application
    ├── detectDigit_fast.py     # Optimized digit detection
    ├── webcam_server.py        # WSL2 webcam streaming server
    ├── requirements.txt        # Python dependencies
    └── retraining/             # Model retraining system
        ├── README.md           # Retraining documentation
        └── scripts/            # Retraining workflow scripts
```

## 🛠️ Development Journey

### Phase 1: Foundation Analysis
- **Code Analysis**: Studied the original `v5.cu` implementation
- **Architecture Review**: Analyzed the 784 → 128 → 10 MLP structure
- **Performance Baseline**: Established baseline accuracy and speed metrics

### Phase 2: Network Enhancement
- **Architecture Expansion**: Redesigned to 784 → 1568 → 784 → 10 for deeper learning
- **Advanced Training**: Implemented decaying learning rate and dropout
- **Weight Management**: Added model saving/loading for persistence
- **Performance Optimization**: CUDA kernel optimization and memory management

### Phase 3: Camera Integration
- **Detection Pipeline**: Developed MSER-based digit detection system
- **Preprocessing**: Created MNIST-compatible image preprocessing
- **Real-Time Processing**: Implemented non-blocking inference pipeline
- **WSL2 Compatibility**: Built webcam streaming solution for Linux development

### Phase 4: Performance Optimization
- **Concurrent Processing**: Implemented multi-threaded inference
- **Smart Caching**: Added hash-based result caching
- **Edge Detection**: Developed border weighting for better center focus
- **Reset System**: Created manual and automatic reset functionality

### Phase 5: Advanced Features
- **Multi-Scale Detection**: Added detection at multiple image scales
- **Nested Detection Removal**: Implemented intelligent duplicate filtering
- **Quality Monitoring**: Added real-time performance metrics
- **Retraining System**: Built comprehensive model fine-tuning workflow

### Phase 6: Webcam-Specific Model Adaptation
- **Data Collection Pipeline**: Automated capture of digit samples from camera stream
- **Interactive Labeling System**: User-friendly interface for manual data annotation
- **Camera-Specific Training**: Fine-tuned model using webcam-collected data
- **Transfer Learning**: Leveraged pre-trained MNIST weights for faster convergence
- **Performance Validation**: Achieved improved accuracy on real-world camera data

## 🚀 Quick Start

### Prerequisites
- CUDA Toolkit (11.0+)
- Python 3.8+
- OpenCV
- WSL2 (for camera development)

### Build and Run
```bash
# Build CUDA executables
make all

# Run inference on test data
make test

# Start camera recognition (requires webcam setup)
make camera

# Run complete retraining workflow
make retrain-workflow
```

### Camera Setup (WSL2)
```bash
# 1. Start webcam server on Windows
python webcam_server.py

# 2. Update IP in camera.py
# 3. Run camera client
python camera.py
```

### Webcam-Specific Model Adaptation
```bash
# 1. Collect samples from your camera setup
make collect-data
# (Write digits in your typical style and lighting)

# 2. Label the collected samples
make label-data
# (Use 0-9 keys to label each digit)

# 3. Retrain model with your data
make retrain-model
# (Model adapts to your specific camera and writing style)

# 4. Test improved accuracy
python3 camera.py
# (Notice better recognition on your setup)
```

## 📊 Performance Achievements

### Neural Network Improvements
- **Accuracy**: Achieved >95% test accuracy on MNIST
- **Architecture**: Enhanced from 2-layer to 4-layer MLP
- **Training**: Implemented advanced optimization techniques
- **Persistence**: Added model weight saving/loading

### Camera Recognition Features
- **Real-Time Processing**: Non-blocking inference pipeline
- **Multi-Threading**: Concurrent detection and inference
- **Smart Detection**: MSER with border weighting and multi-scale processing
- **Reset Capability**: Manual ('r' key) and automatic reset functionality
- **Performance Monitoring**: Real-time FPS and accuracy metrics

## 🎮 Controls

### Camera Application
- **'q'** - Quit application
- **'f'** - Toggle fullscreen mode
- **'r'** - Reset detection state (for repositioned paper)

### Retraining System
- **'q'** - Quit current step
- **'s'** - Save current batch
- **'0-9'** - Label digit during data labeling
- **'n'** - Next sample
- **'p'** - Previous sample

## 🔧 Technical Implementation

### CUDA Optimizations
- **cuBLAS Integration**: Leveraged optimized BLAS operations
- **Memory Management**: Efficient GPU memory allocation
- **Kernel Optimization**: Custom CUDA kernels for specific operations
- **Batch Processing**: Support for multiple simultaneous inferences

### Camera Pipeline
- **MSER Detection**: Robust region detection with stability analysis
- **Preprocessing**: MNIST-compatible image normalization
- **Threading**: Separate threads for capture, detection, and inference
- **Caching**: Hash-based result caching for performance
- **Quality Control**: Border weighting and duplicate removal

### Retraining System
- **Data Collection**: Automated sample capture with quality validation
- **Interactive Labeling**: User-friendly labeling interface
- **Data Organization**: Automatic train/validation/test splitting
- **Model Fine-tuning**: Transfer learning with camera-specific data

## 📈 Results and Metrics

### Neural Network Performance
- **Training Accuracy**: >99% on MNIST training set
- **Test Accuracy**: >95% on MNIST test set
- **Inference Speed**: <1ms per digit on GPU
- **Model Size**: Optimized weight storage and loading

### Camera Recognition Performance
- **Detection Speed**: Real-time processing at 30+ FPS
- **Accuracy**: High accuracy on clear, well-lit digits
- **Robustness**: Handles various lighting conditions and digit sizes
- **Responsiveness**: Immediate feedback with reset capabilities

## 🔮 Future Enhancements

### Neural Network
- **Architecture**: Experiment with convolutional layers
- **Optimization**: Advanced optimization algorithms (Adam, RMSprop)
- **Regularization**: Additional regularization techniques
- **Quantization**: Model quantization for deployment

### Camera System
- **Multi-Digit**: Recognition of multi-digit numbers
- **Handwriting Styles**: Adaptation to different writing styles
- **Real-Time Training**: Online learning from user corrections
- **Mobile Deployment**: Optimization for mobile devices

## 🤝 Contributing

This project builds upon the excellent foundation provided by the original [mnist-cuda](https://github.com/Infatoshi/mnist-cuda) repository. The enhancements focus on:

1. **Architecture Improvements**: Deeper networks with better regularization
2. **Real-World Application**: Camera-based digit recognition
3. **Performance Optimization**: CUDA acceleration and concurrent processing
4. **User Experience**: Interactive retraining and reset capabilities

## 📚 Documentation

- **Camera System**: See `camera/README.md` for detailed camera setup and usage
- **Retraining**: See `camera/retraining/README.md` for model retraining workflow
- **CUDA Implementation**: See `cuda/` directory for neural network source code
- **Build System**: See `Makefile` for available build and run commands

## 🐛 Troubleshooting

### Common Issues
1. **CUDA Compilation**: Ensure CUDA toolkit is properly installed
2. **Camera Connection**: Check webcam server and IP configuration
3. **Model Loading**: Verify model weight files are in correct location
4. **Performance**: Adjust detection parameters for your specific setup

### Performance Tips
- Use good lighting for better digit detection
- Write digits clearly and at reasonable size
- Press 'r' to reset when repositioning paper
- Monitor real-time performance metrics

## 📄 License

This project is based on the original [mnist-cuda](https://github.com/Infatoshi/mnist-cuda) implementation and extends it with significant enhancements for real-world camera-based digit recognition applications.