# Retraining Workflow Makefile
# Run from the project root directory

# Default target
all: help

# Data collection workflow
collect-data:
	@echo "Starting data collection workflow..."
	python3 retraining_workflow.py --step collect

# Complete retraining workflow
retrain-workflow:
	@echo "Starting complete retraining workflow..."
	python3 retraining_workflow.py --step all

# Individual workflow steps
validate-data:
	@echo "Validating collected data..."
	python3 retraining_workflow.py --step validate

label-data:
	@echo "Starting data labeling interface..."
	python3 retraining_workflow.py --step label

organize-data:
	@echo "Organizing data into train/val/test splits..."
	python3 retraining_workflow.py --step organize

retrain-model:
	@echo "Retraining model with collected data..."
	python3 retraining_workflow.py --step retrain

# Workflow status
workflow-status:
	@echo "Checking workflow status..."
	python3 retraining_workflow.py --status

# Individual script usage
collect-only:
	@echo "Running data collection only..."
	python3 data_collection.py

validate-only:
	@echo "Running data validation only..."
	python3 data_validation.py

label-only:
	@echo "Running labeling interface only..."
	python3 labeling_interface.py

organize-only:
	@echo "Running data organization only..."
	python3 data_organizer.py

# Help
help:
	@echo "Retraining Workflow Commands:"
	@echo "  collect-data     - Start data collection from camera"
	@echo "  validate-data    - Validate collected data quality"
	@echo "  label-data       - Start interactive labeling interface"
	@echo "  organize-data    - Split data into train/val/test"
	@echo "  retrain-model    - Retrain model with collected data"
	@echo "  retrain-workflow - Run complete retraining workflow"
	@echo "  workflow-status  - Check current workflow status"
	@echo ""
	@echo "Individual Script Commands:"
	@echo "  collect-only     - Run data collection script directly"
	@echo "  validate-only    - Run data validation script directly"
	@echo "  label-only       - Run labeling interface directly"
	@echo "  organize-only    - Run data organization script directly"
	@echo ""
	@echo "Usage:"
	@echo "  From project root: make -C camera/retraining/scripts <command>"
	@echo "  From this directory: make <command>"

.PHONY: all collect-data validate-data label-data organize-data retrain-model retrain-workflow workflow-status collect-only validate-only label-only organize-only help
