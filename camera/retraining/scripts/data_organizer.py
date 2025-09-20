#!/usr/bin/env python3
"""
Data Organization Script for Collected Digit Samples
Organizes labeled samples into training/validation splits for retraining
"""

import os
import json
import shutil
import argparse
import numpy as np
from collections import defaultdict
import random
from datetime import datetime

class DataOrganizer:
    def __init__(self, data_dir="../data", output_dir="../data"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        
        # Directory structure
        self.train_dir = os.path.join(output_dir, "train")
        self.val_dir = os.path.join(output_dir, "val")
        self.test_dir = os.path.join(output_dir, "test")
        
        # Create output directories
        for split in [self.train_dir, self.val_dir, self.test_dir]:
            os.makedirs(split, exist_ok=True)
            # Create subdirectories for each digit
            for digit in range(10):
                os.makedirs(os.path.join(split, str(digit)), exist_ok=True)
        
        # Statistics
        self.stats = {
            'total_samples': 0,
            'labeled_samples': 0,
            'digit_counts': defaultdict(int),
            'split_counts': {'train': 0, 'val': 0, 'test': 0},
            'split_digit_counts': {
                'train': defaultdict(int),
                'val': defaultdict(int),
                'test': defaultdict(int)
            }
        }
        
        print(f"Data organizer initialized")
        print(f"Source directory: {self.data_dir}")
        print(f"Output directory: {self.output_dir}")
    
    def organize_data(self, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, min_samples_per_digit=5):
        """Organize labeled samples into train/val/test splits"""
        print(f"\n=== Organizing Data ===")
        print(f"Split ratios - Train: {train_ratio}, Val: {val_ratio}, Test: {test_ratio}")
        print(f"Minimum samples per digit: {min_samples_per_digit}")
        
        # Load all labeled samples
        labeled_samples = self._load_labeled_samples()
        
        if not labeled_samples:
            print("No labeled samples found!")
            return False
        
        # Group samples by digit
        digit_groups = defaultdict(list)
        for sample in labeled_samples:
            digit = sample['label']
            digit_groups[digit].append(sample)
        
        # Check minimum samples requirement
        insufficient_digits = []
        for digit, samples in digit_groups.items():
            if len(samples) < min_samples_per_digit:
                insufficient_digits.append((digit, len(samples)))
        
        if insufficient_digits:
            print(f"\nWarning: Some digits have insufficient samples:")
            for digit, count in insufficient_digits:
                print(f"  Digit {digit}: {count} samples (minimum: {min_samples_per_digit})")
            
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                return False
        
        # Organize each digit group
        for digit, samples in digit_groups.items():
            print(f"\nProcessing digit {digit}: {len(samples)} samples")
            self._split_digit_samples(digit, samples, train_ratio, val_ratio, test_ratio)
        
        # Save organization metadata
        self._save_organization_metadata()
        
        # Print final statistics
        self._print_organization_stats()
        
        return True
    
    def _load_labeled_samples(self):
        """Load all labeled samples from metadata directory"""
        metadata_dir = os.path.join(self.data_dir, "metadata")
        labeled_samples = []
        
        if not os.path.exists(metadata_dir):
            print(f"Metadata directory not found: {metadata_dir}")
            return labeled_samples
        
        for filename in os.listdir(metadata_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(metadata_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        metadata = json.load(f)
                    
                    # Only include labeled samples that aren't skipped
                    if (metadata.get('label') is not None and 
                        not metadata.get('skipped', False)):
                        labeled_samples.append(metadata)
                        self.stats['total_samples'] += 1
                        self.stats['digit_counts'][metadata['label']] += 1
                
                except Exception as e:
                    print(f"Error loading {filepath}: {e}")
        
        self.stats['labeled_samples'] = len(labeled_samples)
        print(f"Loaded {len(labeled_samples)} labeled samples")
        
        return labeled_samples
    
    def _split_digit_samples(self, digit, samples, train_ratio, val_ratio, test_ratio):
        """Split samples for a specific digit into train/val/test"""
        # Shuffle samples for random distribution
        random.shuffle(samples)
        
        # Calculate split sizes
        total_samples = len(samples)
        train_size = int(total_samples * train_ratio)
        val_size = int(total_samples * val_ratio)
        test_size = total_samples - train_size - val_size  # Remaining samples go to test
        
        # Split samples
        train_samples = samples[:train_size]
        val_samples = samples[train_size:train_size + val_size]
        test_samples = samples[train_size + val_size:]
        
        # Copy samples to appropriate directories
        self._copy_samples_to_split(train_samples, digit, 'train')
        self._copy_samples_to_split(val_samples, digit, 'val')
        self._copy_samples_to_split(test_samples, digit, 'test')
        
        # Update statistics
        self.stats['split_digit_counts']['train'][digit] = len(train_samples)
        self.stats['split_digit_counts']['val'][digit] = len(val_samples)
        self.stats['split_digit_counts']['test'][digit] = len(test_samples)
        self.stats['split_counts']['train'] += len(train_samples)
        self.stats['split_counts']['val'] += len(val_samples)
        self.stats['split_counts']['test'] += len(test_samples)
        
        print(f"  Train: {len(train_samples)} samples")
        print(f"  Val: {len(val_samples)} samples")
        print(f"  Test: {len(test_samples)} samples")
    
    def _copy_samples_to_split(self, samples, digit, split_name):
        """Copy samples to the appropriate split directory"""
        split_dir = os.path.join(self.output_dir, split_name, str(digit))
        
        for sample in samples:
            sample_id = sample['sample_id']
            
            # Copy preprocessed binary file
            src_bin = os.path.join(self.data_dir, "preprocessed", f"{sample_id}.bin")
            dst_bin = os.path.join(split_dir, f"{sample_id}.bin")
            
            if os.path.exists(src_bin):
                shutil.copy2(src_bin, dst_bin)
            else:
                print(f"Warning: Preprocessed file not found: {src_bin}")
            
            # Copy raw image file
            src_img = os.path.join(self.data_dir, "raw_samples", f"{sample_id}.jpg")
            dst_img = os.path.join(split_dir, f"{sample_id}.jpg")
            
            if os.path.exists(src_img):
                shutil.copy2(src_img, dst_img)
            else:
                print(f"Warning: Raw image not found: {src_img}")
            
            # Copy metadata
            src_meta = os.path.join(self.data_dir, "metadata", f"{sample_id}.json")
            dst_meta = os.path.join(split_dir, f"{sample_id}.json")
            
            if os.path.exists(src_meta):
                shutil.copy2(src_meta, dst_meta)
            else:
                print(f"Warning: Metadata not found: {src_meta}")
    
    def _save_organization_metadata(self):
        """Save organization metadata and statistics"""
        organization_metadata = {
            'organization_timestamp': datetime.now().isoformat(),
            'source_directory': self.data_dir,
            'output_directory': self.output_dir,
            'statistics': dict(self.stats),
            'directory_structure': {
                'train': 'Training data (70% by default)',
                'val': 'Validation data (20% by default)',
                'test': 'Test data (10% by default)'
            }
        }
        
        metadata_file = os.path.join(self.output_dir, "organization_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(organization_metadata, f, indent=2)
        
        print(f"\nOrganization metadata saved to {metadata_file}")
    
    def _print_organization_stats(self):
        """Print final organization statistics"""
        print(f"\n=== Organization Complete ===")
        print(f"Total samples organized: {self.stats['labeled_samples']}")
        print(f"Training samples: {self.stats['split_counts']['train']}")
        print(f"Validation samples: {self.stats['split_counts']['val']}")
        print(f"Test samples: {self.stats['split_counts']['test']}")
        
        print(f"\nDigit distribution by split:")
        for split in ['train', 'val', 'test']:
            print(f"\n{split.upper()}:")
            for digit in range(10):
                count = self.stats['split_digit_counts'][split][digit]
                if count > 0:
                    print(f"  Digit {digit}: {count} samples")
        
        print(f"\nOutput directory: {self.output_dir}")
        print("=" * 40)
    
    def create_training_manifest(self):
        """Create a manifest file for easy training script access"""
        manifest = {
            'created': datetime.now().isoformat(),
            'data_directory': self.output_dir,
            'splits': {}
        }
        
        for split in ['train', 'val', 'test']:
            split_dir = os.path.join(self.output_dir, split)
            split_data = {}
            
            for digit in range(10):
                digit_dir = os.path.join(split_dir, str(digit))
                if os.path.exists(digit_dir):
                    # Count binary files
                    bin_files = [f for f in os.listdir(digit_dir) if f.endswith('.bin')]
                    if bin_files:
                        split_data[str(digit)] = {
                            'count': len(bin_files),
                            'files': bin_files
                        }
            
            if split_data:
                manifest['splits'][split] = split_data
        
        manifest_file = os.path.join(self.output_dir, "training_manifest.json")
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"Training manifest created: {manifest_file}")
        return manifest_file

def main():
    parser = argparse.ArgumentParser(description='Organize collected digit samples for training')
    parser.add_argument('--data-dir', default='../data',
                       help='Directory containing collected data')
    parser.add_argument('--output-dir', default='../data',
                       help='Output directory for organized training data')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                       help='Ratio of data for training (default: 0.7)')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                       help='Ratio of data for validation (default: 0.2)')
    parser.add_argument('--test-ratio', type=float, default=0.1,
                       help='Ratio of data for testing (default: 0.1)')
    parser.add_argument('--min-samples', type=int, default=5,
                       help='Minimum samples per digit (default: 5)')
    parser.add_argument('--create-manifest', action='store_true',
                       help='Create training manifest file')
    
    args = parser.parse_args()
    
    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        print(f"Error: Ratios must sum to 1.0, got {total_ratio}")
        return
    
    # Create data organizer
    organizer = DataOrganizer(
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
    
    # Organize data
    success = organizer.organize_data(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        min_samples_per_digit=args.min_samples
    )
    
    if success and args.create_manifest:
        organizer.create_training_manifest()

if __name__ == "__main__":
    main()
