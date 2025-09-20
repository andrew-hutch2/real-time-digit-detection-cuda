# CUDA compiler
NVCC = nvcc

# Compiler flags
NVCCFLAGS = -O3 -lcublas -lcurand

# Directories
CUDA_DIR = cuda
BIN_DIR = bin
DATA_DIR = ../data

# Targets
TRAIN_EXEC = $(BIN_DIR)/train2
INFERENCE_EXEC = $(BIN_DIR)/inference
RETRAIN_EXEC = $(BIN_DIR)/retrain
TEST_SCRIPT = test_inference.py

# Default target
all: $(TRAIN_EXEC) $(INFERENCE_EXEC) $(RETRAIN_EXEC)

# Create bin directory
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Train the model
$(TRAIN_EXEC): $(CUDA_DIR)/train2.cu $(CUDA_DIR)/modelsave.cu $(CUDA_DIR)/neuralNetwork.cuh | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $(CUDA_DIR)/train2.cu $(CUDA_DIR)/modelsave.cu

# Inference program
$(INFERENCE_EXEC): $(CUDA_DIR)/inference.cu $(CUDA_DIR)/neuralNetwork.cuh | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $(CUDA_DIR)/inference.cu

# Retraining program
$(RETRAIN_EXEC): $(CUDA_DIR)/retrain.cu $(CUDA_DIR)/modelsave.cu $(CUDA_DIR)/neuralNetwork.cu $(CUDA_DIR)/neuralNetwork.cuh | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $(CUDA_DIR)/retrain.cu $(CUDA_DIR)/modelsave.cu $(CUDA_DIR)/neuralNetwork.cu

# Train the model
train: $(TRAIN_EXEC)
	@echo "Training the model..."
	./$(TRAIN_EXEC)

# Test inference with generated images
test: $(INFERENCE_EXEC)
	@echo "Generating test images and testing inference..."
	python3 $(TEST_SCRIPT)

# Test all digits
test-all: $(INFERENCE_EXEC)
	@echo "Testing all digits..."
	./test_all_digits.sh

# Clean compiled files
clean:
	rm -rf $(BIN_DIR)/
	rm -rf test_images/
	rm -f *.bin

# Install dependencies (Ubuntu/Debian)
install-deps:
	@echo "Installing dependencies..."
	sudo apt-get update
	sudo apt-get install -y nvidia-cuda-toolkit python3-numpy python3-pip
	pip3 install torch torchvision

# Camera pipeline setup
setup-camera:
	@echo "Setting up camera pipeline..."
	cd camera && ./setup_camera.sh

# Test camera pipeline
test-camera: $(INFERENCE_EXEC)
	@echo "Testing camera pipeline..."
	cd camera && python3 test_image_generator.py
	cd camera && python3 webcam_processor.py test_images/single_digit_5.jpg

# Data collection workflow
collect-data:
	@echo "Starting data collection workflow..."
	$(MAKE) -C camera/retraining collect-data

# Complete retraining workflow
retrain-workflow:
	@echo "Starting complete retraining workflow..."
	$(MAKE) -C camera/retraining retrain-workflow

# Individual workflow steps
validate-data:
	@echo "Validating collected data..."
	$(MAKE) -C camera/retraining validate-data

label-data:
	@echo "Starting data labeling interface..."
	$(MAKE) -C camera/retraining label-data

organize-data:
	@echo "Organizing data into train/val/test splits..."
	$(MAKE) -C camera/retraining organize-data

retrain-model: $(RETRAIN_EXEC)
	@echo "Retraining model with collected data..."
	$(MAKE) -C camera/retraining retrain-model

# Workflow status
workflow-status:
	@echo "Checking workflow status..."
	$(MAKE) -C camera/retraining workflow-status

# Help
help:
	@echo "Available targets:"
	@echo "  all        - Compile training, inference, and retraining programs"
	@echo "  train      - Train the model and save weights"
	@echo "  test       - Generate test images and test inference"
	@echo "  test-all   - Test all digits with the trained model"
	@echo "  setup-camera - Set up camera pipeline for webcam processing"
	@echo "  test-camera - Test camera pipeline with synthetic images"
	@echo ""
	@echo "Data Collection and Retraining:"
	@echo "  collect-data - Start data collection from camera"
	@echo "  validate-data - Validate collected data quality"
	@echo "  label-data  - Start interactive labeling interface"
	@echo "  organize-data - Organize data into train/val/test splits"
	@echo "  retrain-model - Retrain model with collected data"
	@echo "  retrain-workflow - Run complete retraining workflow"
	@echo "  workflow-status - Check current workflow status"
	@echo ""
	@echo "Utilities:"
	@echo "  clean      - Remove compiled executables and test files"
	@echo "  install-deps - Install required dependencies"
	@echo "  help       - Show this help message"

.PHONY: all train test test-all setup-camera test-camera collect-data validate-data label-data organize-data retrain-model retrain-workflow workflow-status clean install-deps help
