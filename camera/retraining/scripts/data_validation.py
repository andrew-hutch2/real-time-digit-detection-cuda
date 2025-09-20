#!/usr/bin/env python3
"""
Data Validation Script for Collected Digit Samples
Validates quality and consistency of collected training data
"""

import os
import json
import numpy as np
import cv2
import argparse
import glob
from collections import defaultdict
import matplotlib.pyplot as plt

class DataValidator:
    def __init__(self, data_dir="../data"):
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, "raw_samples")
        self.preprocessed_dir = os.path.join(data_dir, "preprocessed")
        self.metadata_dir = os.path.join(data_dir, "metadata")
        
        # Validation results
        self.validation_results = {
            'total_samples': 0,
            'valid_samples': 0,
            'invalid_samples': 0,
            'issues': defaultdict(list),
            'digit_distribution': defaultdict(int),
            'quality_metrics': {}
        }
        
        print(f"Data validator initialized for: {data_dir}")
    
    def validate_all_data(self):
        """Run comprehensive validation on all collected data"""
        print("\n=== Starting Data Validation ===")
        
        # Check directory structure
        self._validate_directory_structure()
        
        # Validate individual samples
        self._validate_samples()
        
        # Analyze data quality
        self._analyze_data_quality()
        
        # Generate validation report
        self._generate_validation_report()
        
        print("\n=== Validation Complete ===")
        self._print_validation_summary()
    
    def _validate_directory_structure(self):
        """Validate that required directories exist"""
        required_dirs = [self.raw_dir, self.preprocessed_dir, self.metadata_dir]
        
        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                self.validation_results['issues']['missing_directories'].append(dir_path)
                print(f"ERROR: Missing directory: {dir_path}")
            else:
                print(f"✓ Directory exists: {dir_path}")
    
    def _validate_samples(self):
        """Validate individual samples"""
        metadata_files = glob.glob(os.path.join(self.metadata_dir, "*.json"))
        self.validation_results['total_samples'] = len(metadata_files)
        
        print(f"\nValidating {len(metadata_files)} samples...")
        
        for metadata_file in metadata_files:
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                sample_id = metadata['sample_id']
                is_valid = True
                
                # Check if all required files exist
                raw_file = metadata.get('raw_file')
                # Resolve relative paths relative to the data directory
                if raw_file and not os.path.isabs(raw_file):
                    raw_file = os.path.join(self.data_dir, raw_file)
                preprocessed_file = os.path.join(self.preprocessed_dir, f"{sample_id}.bin")
                
                if not os.path.exists(raw_file):
                    self.validation_results['issues']['missing_raw_files'].append(sample_id)
                    is_valid = False
                
                if not os.path.exists(preprocessed_file):
                    self.validation_results['issues']['missing_preprocessed_files'].append(sample_id)
                    is_valid = False
                
                # Validate preprocessed data
                if os.path.exists(preprocessed_file):
                    if not self._validate_preprocessed_data(preprocessed_file, sample_id):
                        is_valid = False
                
                # Validate raw image
                if os.path.exists(raw_file):
                    if not self._validate_raw_image(raw_file, sample_id):
                        is_valid = False
                
                # Check metadata completeness
                if not self._validate_metadata(metadata, sample_id):
                    is_valid = False
                
                # Update statistics
                if is_valid:
                    self.validation_results['valid_samples'] += 1
                    if metadata.get('label') is not None:
                        self.validation_results['digit_distribution'][metadata['label']] += 1
                else:
                    self.validation_results['invalid_samples'] += 1
                
            except Exception as e:
                self.validation_results['issues']['metadata_errors'].append(f"{metadata_file}: {str(e)}")
                self.validation_results['invalid_samples'] += 1
    
    def _validate_preprocessed_data(self, filepath, sample_id):
        """Validate preprocessed binary data"""
        try:
            # Load data
            data = np.fromfile(filepath, dtype=np.float32)
            
            # Check size
            if len(data) != 784:  # 28x28 = 784
                self.validation_results['issues']['incorrect_preprocessed_size'].append(
                    f"{sample_id}: expected 784, got {len(data)}")
                return False
            
            # Check for NaN or infinite values
            if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                self.validation_results['issues']['invalid_preprocessed_values'].append(sample_id)
                return False
            
            # Check value range (should be normalized)
            if np.min(data) < -5.0 or np.max(data) > 5.0:
                self.validation_results['issues']['preprocessed_out_of_range'].append(sample_id)
                return False
            
            return True
            
        except Exception as e:
            self.validation_results['issues']['preprocessed_errors'].append(f"{sample_id}: {str(e)}")
            return False
    
    def _validate_raw_image(self, filepath, sample_id):
        """Validate raw image data"""
        try:
            # Load image
            img = cv2.imread(filepath)
            
            if img is None:
                self.validation_results['issues']['corrupt_raw_images'].append(sample_id)
                return False
            
            # Check image dimensions
            h, w = img.shape[:2]
            if h < 10 or w < 10 or h > 500 or w > 500:
                self.validation_results['issues']['invalid_raw_dimensions'].append(
                    f"{sample_id}: {w}x{h}")
                return False
            
            return True
            
        except Exception as e:
            self.validation_results['issues']['raw_image_errors'].append(f"{sample_id}: {str(e)}")
            return False
    
    def _validate_metadata(self, metadata, sample_id):
        """Validate metadata completeness"""
        required_fields = ['sample_id', 'timestamp', 'bbox', 'preprocessed_shape']
        
        for field in required_fields:
            if field not in metadata:
                self.validation_results['issues']['missing_metadata_fields'].append(
                    f"{sample_id}: missing {field}")
                return False
        
        # Validate bbox format
        bbox = metadata.get('bbox')
        if not isinstance(bbox, list) or len(bbox) != 4:
            self.validation_results['issues']['invalid_bbox_format'].append(sample_id)
            return False
        
        return True
    
    def _analyze_data_quality(self):
        """Analyze overall data quality metrics"""
        print("\nAnalyzing data quality...")
        
        # Calculate quality metrics
        total_samples = self.validation_results['total_samples']
        valid_samples = self.validation_results['valid_samples']
        
        if total_samples > 0:
            self.validation_results['quality_metrics'] = {
                'validity_rate': valid_samples / total_samples,
                'total_samples': total_samples,
                'valid_samples': valid_samples,
                'invalid_samples': self.validation_results['invalid_samples'],
                'digit_coverage': len([d for d in self.validation_results['digit_distribution'].values() if d > 0]),
                'min_samples_per_digit': min(self.validation_results['digit_distribution'].values()) if self.validation_results['digit_distribution'] else 0,
                'max_samples_per_digit': max(self.validation_results['digit_distribution'].values()) if self.validation_results['digit_distribution'] else 0
            }
    
    def _generate_validation_report(self):
        """Generate detailed validation report"""
        report_file = os.path.join(self.data_dir, "validation_report.json")
        
        with open(report_file, 'w') as f:
            json.dump(self.validation_results, f, indent=2)
        
        print(f"Validation report saved to: {report_file}")
    
    def _print_validation_summary(self):
        """Print validation summary"""
        metrics = self.validation_results['quality_metrics']
        
        if metrics:
            print(f"Total samples: {metrics.get('total_samples', 0)}")
            print(f"Valid samples: {metrics.get('valid_samples', 0)}")
            print(f"Invalid samples: {metrics.get('invalid_samples', 0)}")
            print(f"Validity rate: {metrics.get('validity_rate', 0)*100:.1f}%")
            print(f"Digit coverage: {metrics.get('digit_coverage', 0)}/10 digits")
            print(f"Min samples per digit: {metrics.get('min_samples_per_digit', 0)}")
            print(f"Max samples per digit: {metrics.get('max_samples_per_digit', 0)}")
        else:
            print("No validation metrics available")
        
        if self.validation_results['issues']:
            print(f"\nIssues found:")
            for issue_type, issues in self.validation_results['issues'].items():
                if issues:
                    print(f"  {issue_type}: {len(issues)} issues")
                    if len(issues) <= 5:  # Show first few issues
                        for issue in issues[:5]:
                            print(f"    - {issue}")
                    else:
                        print(f"    - {issues[0]} (and {len(issues)-1} more...)")
    
    def visualize_data_distribution(self, save_plot=True):
        """Create visualization of data distribution"""
        if not self.validation_results['digit_distribution']:
            print("No digit distribution data available")
            return
        
        digits = list(range(10))
        counts = [self.validation_results['digit_distribution'][d] for d in digits]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(digits, counts, color='skyblue', edgecolor='navy', alpha=0.7)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            if count > 0:
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        str(count), ha='center', va='bottom')
        
        plt.xlabel('Digit')
        plt.ylabel('Number of Samples')
        plt.title('Distribution of Collected Digit Samples')
        plt.xticks(digits)
        plt.grid(axis='y', alpha=0.3)
        
        if save_plot:
            plot_file = os.path.join(self.data_dir, "data_distribution.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"Distribution plot saved to: {plot_file}")
        
        plt.show()
    
    def clean_invalid_data(self, dry_run=True):
        """Remove invalid samples from the dataset"""
        if dry_run:
            print("DRY RUN: Would remove the following invalid samples:")
        else:
            print("Removing invalid samples...")
        
        removed_count = 0
        
        # Remove samples with missing files
        for sample_id in self.validation_results['issues']['missing_raw_files']:
            if not dry_run:
                self._remove_sample(sample_id)
            print(f"  - {sample_id} (missing raw file)")
            removed_count += 1
        
        for sample_id in self.validation_results['issues']['missing_preprocessed_files']:
            if not dry_run:
                self._remove_sample(sample_id)
            print(f"  - {sample_id} (missing preprocessed file)")
            removed_count += 1
        
        # Remove samples with invalid data
        for sample_id in self.validation_results['issues']['invalid_preprocessed_values']:
            if not dry_run:
                self._remove_sample(sample_id)
            print(f"  - {sample_id} (invalid preprocessed values)")
            removed_count += 1
        
        for sample_id in self.validation_results['issues']['corrupt_raw_images']:
            if not dry_run:
                self._remove_sample(sample_id)
            print(f"  - {sample_id} (corrupt raw image)")
            removed_count += 1
        
        if dry_run:
            print(f"\nDRY RUN: Would remove {removed_count} invalid samples")
            print("Run with --clean to actually remove them")
        else:
            print(f"\nRemoved {removed_count} invalid samples")
    
    def _remove_sample(self, sample_id):
        """Remove a sample and all its associated files"""
        # Remove metadata
        metadata_file = os.path.join(self.metadata_dir, f"{sample_id}.json")
        if os.path.exists(metadata_file):
            os.remove(metadata_file)
        
        # Remove raw image
        raw_file = os.path.join(self.raw_dir, f"{sample_id}.jpg")
        if os.path.exists(raw_file):
            os.remove(raw_file)
        
        # Remove preprocessed data
        preprocessed_file = os.path.join(self.preprocessed_dir, f"{sample_id}.bin")
        if os.path.exists(preprocessed_file):
            os.remove(preprocessed_file)

def main():
    parser = argparse.ArgumentParser(description='Validate collected digit samples')
    parser.add_argument('--data-dir', default='../data',
                       help='Directory containing collected data')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualization of data distribution')
    parser.add_argument('--clean', action='store_true',
                       help='Remove invalid samples (use with caution)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be cleaned without actually removing files')
    
    args = parser.parse_args()
    
    # Create validator
    validator = DataValidator(data_dir=args.data_dir)
    
    # Run validation
    validator.validate_all_data()
    
    # Create visualization if requested
    if args.visualize:
        validator.visualize_data_distribution()
    
    # Clean invalid data if requested
    if args.clean or args.dry_run:
        validator.clean_invalid_data(dry_run=args.dry_run)

if __name__ == "__main__":
    main()
