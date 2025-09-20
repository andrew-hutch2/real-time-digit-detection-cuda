#!/usr/bin/env python3
"""
Complete Retraining Workflow for Camera-Based Digit Recognition
Automates the entire process from data collection to model retraining
"""

import os
import sys
import subprocess
import argparse
import time
from datetime import datetime

class RetrainingWorkflow:
    def __init__(self, base_dir="../../.."):
        self.base_dir = base_dir
        self.camera_dir = "camera"
        self.cuda_dir = "cuda"
        self.bin_dir = "bin"
        
        # Workflow directories
        self.data_dir = os.path.join(self.camera_dir, "retraining", "data")
        self.training_dir = os.path.join(self.camera_dir, "retraining", "data")
        
        # Default parameters
        self.target_samples = 200
        self.train_ratio = 0.7
        self.val_ratio = 0.2
        self.test_ratio = 0.1
        self.epochs = 150
        self.learning_rate = 0.001
        
        print("Retraining Workflow Initialized")
        print(f"Base directory: {self.base_dir}")
        print(f"Data directory: {self.data_dir}")
        print(f"Training directory: {self.training_dir}")
    
    def run_complete_workflow(self, skip_collection=False, skip_labeling=False, 
                            skip_validation=False, skip_organization=False, 
                            skip_retraining=False):
        """Run the complete retraining workflow"""
        print("\n" + "="*60)
        print("STARTING COMPLETE RETRAINING WORKFLOW")
        print("="*60)
        
        workflow_start = time.time()
        
        try:
            # Step 1: Data Collection
            if not skip_collection:
                print("\n[STEP 1] Data Collection")
                print("-" * 30)
                success = self.collect_data()
                if success:
                    print("\nData collection completed!")
                    print("You can now proceed to labeling or run the complete workflow.")
                    response = input("Continue to labeling? (y/n): ")
                    if response.lower() != 'y':
                        print("Stopping workflow. You can resume later with: make label-data")
                        return True
            else:
                print("\n[STEP 1] Data Collection - SKIPPED")
            
            # Step 2: Data Validation
            if not skip_validation:
                print("\n[STEP 2] Data Validation")
                print("-" * 30)
                self.validate_data()
            else:
                print("\n[STEP 2] Data Validation - SKIPPED")
            
            # Step 3: Data Labeling
            if not skip_labeling:
                print("\n[STEP 3] Data Labeling")
                print("-" * 30)
                self.label_data()
            else:
                print("\n[STEP 3] Data Labeling - SKIPPED")
            
            # Step 4: Data Organization
            if not skip_organization:
                print("\n[STEP 4] Data Organization")
                print("-" * 30)
                self.organize_data()
            else:
                print("\n[STEP 4] Data Organization - SKIPPED")
            
            # Step 5: Model Retraining
            if not skip_retraining:
                print("\n[STEP 5] Model Retraining")
                print("-" * 30)
                self.retrain_model()
            else:
                print("\n[STEP 5] Model Retraining - SKIPPED")
            
            # Workflow complete
            workflow_duration = time.time() - workflow_start
            print(f"\n" + "="*60)
            print("WORKFLOW COMPLETE!")
            print(f"Total duration: {workflow_duration:.1f} seconds")
            print("="*60)
            
        except KeyboardInterrupt:
            print("\n\nWorkflow interrupted by user")
            return False
        except Exception as e:
            print(f"\n\nWorkflow failed with error: {e}")
            return False
        
        return True
    
    def collect_data(self):
        """Step 1: Collect digit samples from camera"""
        print("Starting data collection...")
        
        # Check if data collection script exists
        script_path = "data_collection.py"
        if not os.path.exists(script_path):
            print(f"ERROR: Data collection script not found: {script_path}")
            return False
        
        # Run data collection
        cmd = [
            "python3", script_path,
            "--output-dir", self.data_dir,
            "--target-samples", str(self.target_samples)
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        
        if result.returncode == 0:
            print("✓ Data collection completed successfully")
            return True
        else:
            print("✗ Data collection failed")
            return False
    
    def validate_data(self):
        """Step 2: Validate collected data"""
        print("Starting data validation...")
        
        # Check if validation script exists
        script_path = "data_validation.py"
        if not os.path.exists(script_path):
            print(f"ERROR: Data validation script not found: {script_path}")
            return False
        
        # Run data validation
        cmd = [
            "python3", script_path,
            "--data-dir", self.data_dir,
            "--visualize"
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        
        if result.returncode == 0:
            print("✓ Data validation completed successfully")
            return True
        else:
            print("✗ Data validation failed")
            return False
    
    def label_data(self):
        """Step 3: Label collected data"""
        print("Starting data labeling...")
        print("This will open an interactive labeling interface.")
        print("Please label the collected samples manually.")
        
        # Check if labeling script exists
        script_path = "labeling_interface.py"
        if not os.path.exists(script_path):
            print(f"ERROR: Labeling script not found: {script_path}")
            return False
        
        # Run labeling interface
        cmd = [
            "python3", script_path,
            "--data-dir", self.data_dir
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        
        if result.returncode == 0:
            print("✓ Data labeling completed successfully")
            return True
        else:
            print("✗ Data labeling failed")
            return False
    
    def organize_data(self):
        """Step 4: Organize data into train/val/test splits"""
        print("Starting data organization...")
        
        # Check if organization script exists
        script_path = "data_organizer.py"
        if not os.path.exists(script_path):
            print(f"ERROR: Data organization script not found: {script_path}")
            return False
        
        # Run data organization
        cmd = [
            "python3", script_path,
            "--data-dir", self.data_dir,
            "--output-dir", self.training_dir,
            "--train-ratio", str(self.train_ratio),
            "--val-ratio", str(self.val_ratio),
            "--test-ratio", str(self.test_ratio),
            "--create-manifest"
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        
        if result.returncode == 0:
            print("✓ Data organization completed successfully")
            return True
        else:
            print("✗ Data organization failed")
            return False
    
    def retrain_model(self):
        """Step 5: Retrain the model"""
        print("Starting model retraining...")
        
        # Check if retraining script exists
        script_path = os.path.join(self.base_dir, self.cuda_dir, "retrain.cu")
        modelsave_path = os.path.join(self.base_dir, self.cuda_dir, "modelsave.cu")
        neuralnet_path = os.path.join(self.base_dir, self.cuda_dir, "neuralNetwork.cu")
        
        # Check all required files exist
        required_files = [
            (script_path, "retrain.cu"),
            (modelsave_path, "modelsave.cu"), 
            (neuralnet_path, "neuralNetwork.cu")
        ]
        
        for file_path, name in required_files:
            if not os.path.exists(file_path):
                print(f"ERROR: {name} not found: {file_path}")
                print(f"Looking for: {os.path.abspath(file_path)}")
                return False
            else:
                print(f"✓ Found {name}: {os.path.abspath(file_path)}")
        
        # Compile retraining program
        print("Compiling retraining program...")
        output_path = os.path.abspath(os.path.join(self.base_dir, self.bin_dir, "retrain"))
        compile_cmd = [
            "nvcc", "-O3", "-lcublas", "-lcurand",
            "-o", output_path,
            os.path.abspath(script_path), 
            os.path.abspath(modelsave_path),
            os.path.abspath(neuralnet_path)
        ]
        
        print(f"Running: {' '.join(compile_cmd)}")
        # Resolve the base directory path
        base_dir_abs = os.path.abspath(self.base_dir)
        print(f"Working directory: {base_dir_abs}")
        compile_result = subprocess.run(compile_cmd, cwd=base_dir_abs)
        
        if compile_result.returncode != 0:
            print("✗ Compilation failed")
            return False
        
        print("✓ Compilation successful")
        
        # Run retraining
        retrain_exec = os.path.abspath(os.path.join(self.base_dir, self.bin_dir, "retrain"))
        base_weights = os.path.abspath(os.path.join(self.base_dir, self.bin_dir, "trained_model_weights.bin"))
        
        cmd = [
            retrain_exec,
            os.path.abspath(os.path.join(self.base_dir, self.training_dir)),
            base_weights,
            str(self.epochs),
            str(self.learning_rate)
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=self.base_dir)
        
        if result.returncode == 0:
            print("✓ Model retraining completed successfully")
            print("Retrained model saved as: retrained_model_best.bin")
            return True
        else:
            print("✗ Model retraining failed")
            return False
    
    def run_individual_step(self, step):
        """Run an individual workflow step"""
        if step == "collect":
            return self.collect_data()
        elif step == "validate":
            return self.validate_data()
        elif step == "label":
            return self.label_data()
        elif step == "organize":
            return self.organize_data()
        elif step == "retrain":
            return self.retrain_model()
        else:
            print(f"Unknown step: {step}")
            return False
    
    def print_workflow_status(self):
        """Print current status of the workflow"""
        print("\n=== Workflow Status ===")
        
        # Check data collection
        if os.path.exists(self.data_dir):
            metadata_files = len([f for f in os.listdir(os.path.join(self.data_dir, "metadata")) 
                                if f.endswith('.json')])
            print(f"Data Collection: {metadata_files} samples collected")
        else:
            print("Data Collection: No data collected yet")
        
        # Check data organization
        if os.path.exists(self.training_dir):
            train_files = len([f for f in os.listdir(os.path.join(self.training_dir, "train")) 
                             if os.path.isdir(os.path.join(self.training_dir, "train", f))])
            print(f"Data Organization: {train_files} digit classes organized")
        else:
            print("Data Organization: Not organized yet")
        
        # Check retrained model
        retrained_model = os.path.join(self.bin_dir, "retrained_model_best.bin")
        if os.path.exists(retrained_model):
            print("Model Retraining: Retrained model available")
        else:
            print("Model Retraining: No retrained model found")

def main():
    parser = argparse.ArgumentParser(description='Complete retraining workflow for camera-based digit recognition')
    parser.add_argument('--step', choices=['collect', 'validate', 'label', 'organize', 'retrain', 'all'],
                       default='all', help='Workflow step to run')
    parser.add_argument('--target-samples', type=int, default=200,
                       help='Target number of samples to collect')
    parser.add_argument('--epochs', type=int, default=250,
                       help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate for retraining')
    parser.add_argument('--skip-collection', action='store_true',
                       help='Skip data collection step')
    parser.add_argument('--skip-labeling', action='store_true',
                       help='Skip data labeling step')
    parser.add_argument('--skip-validation', action='store_true',
                       help='Skip data validation step')
    parser.add_argument('--skip-organization', action='store_true',
                       help='Skip data organization step')
    parser.add_argument('--skip-retraining', action='store_true',
                       help='Skip model retraining step')
    parser.add_argument('--status', action='store_true',
                       help='Show current workflow status')
    
    args = parser.parse_args()
    
    # Create workflow
    workflow = RetrainingWorkflow()
    
    # Set parameters
    workflow.target_samples = args.target_samples
    workflow.epochs = args.epochs
    workflow.learning_rate = args.learning_rate
    
    # Show status if requested
    if args.status:
        workflow.print_workflow_status()
        return
    
    # Run workflow
    if args.step == 'all':
        success = workflow.run_complete_workflow(
            skip_collection=args.skip_collection,
            skip_labeling=args.skip_labeling,
            skip_validation=args.skip_validation,
            skip_organization=args.skip_organization,
            skip_retraining=args.skip_retraining
        )
    else:
        success = workflow.run_individual_step(args.step)
    
    if success:
        print("\nWorkflow completed successfully!")
    else:
        print("\nWorkflow failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
