#!/usr/bin/env python3
"""
Labeling Interface for Collected Digit Samples
Interactive tool for manually labeling collected digit samples
"""

import cv2
import numpy as np
import os
import json
import argparse
import glob
from datetime import datetime

class DigitLabelingInterface:
    def __init__(self, data_dir="../data"):
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, "raw_samples")
        self.preprocessed_dir = os.path.join(data_dir, "preprocessed")
        self.metadata_dir = os.path.join(data_dir, "metadata")
        
        # Load all unlabeled samples
        self.samples = self._load_samples()
        self.current_index = 0
        self.labeled_count = 0
        
        # Labeling statistics
        self.stats = {
            'total_samples': len(self.samples),
            'labeled_samples': 0,
            'digit_counts': {i: 0 for i in range(10)},
            'session_start': datetime.now().isoformat()
        }
        
        print(f"Loaded {len(self.samples)} samples for labeling")
        print("Controls:")
        print("  0-9: Label digit")
        print("  n: Next sample")
        print("  p: Previous sample")
        print("  s: Skip sample")
        print("  d: Delete current sample")
        print("  q: Quit and save")
        print("  r: Reset current sample")
        print("=" * 40)
    
    def _load_samples(self):
        """Load all samples that need labeling"""
        samples = []
        
        # Get all metadata files
        metadata_files = glob.glob(os.path.join(self.metadata_dir, "*.json"))
        
        for metadata_file in metadata_files:
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # Only include unlabeled samples
                if metadata.get('label') is None:
                    samples.append(metadata)
            except Exception as e:
                print(f"Error loading {metadata_file}: {e}")
        
        # Sort by sample_id for consistent ordering
        samples.sort(key=lambda x: x['sample_id'])
        return samples
    
    def start_labeling(self):
        """Start the interactive labeling process"""
        if not self.samples:
            print("No unlabeled samples found!")
            return
        
        while self.current_index < len(self.samples):
            sample = self.samples[self.current_index]
            self._display_sample(sample)
            
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('q'):
                break
            elif key >= ord('0') and key <= ord('9'):
                digit = key - ord('0')
                self._label_sample(sample, digit)
                self._next_sample()
            elif key == ord('n'):
                self._next_sample()
            elif key == ord('p'):
                self._previous_sample()
            elif key == ord('s'):
                self._skip_sample(sample)
                self._next_sample()
            elif key == ord('r'):
                self._reset_sample(sample)
            elif key == ord('d'):
                self._delete_sample(sample)
                # Move to next sample or previous if at end
                if self.current_index >= len(self.samples):
                    self.current_index = len(self.samples) - 1
                if self.current_index >= 0:
                    self._display_sample(self.samples[self.current_index])
                else:
                    print("No more samples to display")
                    break
            else:
                print(f"Unknown key: {chr(key)}")
        
        self._save_labeling_session()
        cv2.destroyAllWindows()
        self._print_final_stats()
    
    def _display_sample(self, sample):
        """Display current sample for labeling"""
        # Load raw image
        raw_file = sample['raw_file']
        # Resolve relative paths relative to the data directory
        if raw_file and not os.path.isabs(raw_file):
            raw_file = os.path.join(self.data_dir, raw_file)
        if os.path.exists(raw_file):
            raw_img = cv2.imread(raw_file)
        else:
            print(f"Raw image not found: {raw_file}")
            return
        
        # Load preprocessed image
        preprocessed_file = sample['preprocessed_file'] = os.path.join(
            self.preprocessed_dir, f"{sample['sample_id']}.bin"
        )
        
        if os.path.exists(preprocessed_file):
            # Load and reshape preprocessed data
            preprocessed_data = np.fromfile(preprocessed_file, dtype=np.float32)
            preprocessed_img = preprocessed_data.reshape(28, 28)
            
            # Denormalize for display (reverse MNIST normalization)
            mean, std = 0.1307, 0.3081
            display_img = (preprocessed_img * std + mean) * 255
            display_img = np.clip(display_img, 0, 255).astype(np.uint8)
            
            # Resize for better visibility
            display_img = cv2.resize(display_img, (280, 280), interpolation=cv2.INTER_NEAREST)
        else:
            print(f"Preprocessed image not found: {preprocessed_file}")
            return
        
        # Create display window
        display = np.zeros((400, 600, 3), dtype=np.uint8)
        
        # Place raw image (left side)
        if raw_img is not None:
            # Resize raw image to fit
            raw_resized = cv2.resize(raw_img, (200, 200))
            display[50:250, 50:250] = raw_resized
        
        # Place preprocessed image (right side)
        if len(display_img.shape) == 2:  # Grayscale
            display_img_bgr = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)
        else:  # Already BGR
            display_img_bgr = display_img
        
        # Resize to fit the display area (250x250)
        display_img_resized = cv2.resize(display_img_bgr, (250, 250))
        display[50:300, 350:600] = display_img_resized
        
        # Add text information
        info_text = [
            f"Sample: {self.current_index + 1}/{len(self.samples)}",
            f"ID: {sample['sample_id']}",
            f"Current Label: {sample.get('label', 'None')}",
            f"Labeled: {self.labeled_count}/{len(self.samples)}"
        ]
        
        for i, text in enumerate(info_text):
            cv2.putText(display, text, (10, 20 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Add digit count statistics
        stats_text = "Digit counts:"
        cv2.putText(display, stats_text, (10, 350), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        for i in range(10):
            count = self.stats['digit_counts'][i]
            text = f"{i}: {count}"
            cv2.putText(display, text, (10 + (i % 5) * 60, 370 + (i // 5) * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        cv2.imshow('Digit Labeling Interface', display)
    
    def _label_sample(self, sample, digit):
        """Label a sample with the given digit"""
        sample['label'] = digit
        sample['labeling_timestamp'] = datetime.now().isoformat()
        sample['labeling_confidence'] = 1.0  # Manual labeling is 100% confident
        
        # Update statistics
        self.stats['digit_counts'][digit] += 1
        self.labeled_count += 1
        
        # Save updated metadata
        metadata_file = os.path.join(self.metadata_dir, f"{sample['sample_id']}.json")
        with open(metadata_file, 'w') as f:
            json.dump(sample, f, indent=2)
        
        print(f"Labeled {sample['sample_id']} as digit {digit}")
    
    def _skip_sample(self, sample):
        """Mark sample as skipped"""
        sample['skipped'] = True
        sample['skip_timestamp'] = datetime.now().isoformat()
        
        # Save updated metadata
        metadata_file = os.path.join(self.metadata_dir, f"{sample['sample_id']}.json")
        with open(metadata_file, 'w') as f:
            json.dump(sample, f, indent=2)
        
        print(f"Skipped {sample['sample_id']}")
    
    def _reset_sample(self, sample):
        """Reset sample label"""
        if 'label' in sample:
            old_label = sample['label']
            del sample['label']
            self.stats['digit_counts'][old_label] -= 1
            self.labeled_count -= 1
        
        # Save updated metadata
        metadata_file = os.path.join(self.metadata_dir, f"{sample['sample_id']}.json")
        with open(metadata_file, 'w') as f:
            json.dump(sample, f, indent=2)
        
        print(f"Reset {sample['sample_id']}")
    
    def _delete_sample(self, sample):
        """Delete current sample and its files"""
        sample_id = sample['sample_id']
        
        # Remove from samples list
        self.samples.pop(self.current_index)
        
        # Delete files
        raw_file = sample['raw_file']
        # Resolve relative paths relative to the data directory
        if raw_file and not os.path.isabs(raw_file):
            raw_file = os.path.join(self.data_dir, raw_file)
        
        files_to_delete = [
            raw_file,
            sample['preprocessed_file'],
            os.path.join(self.metadata_dir, f"{sample_id}.json")
        ]
        
        for file_path in files_to_delete:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except OSError as e:
                    print(f"Error deleting {file_path}: {e}")
        
        print(f"Deleted sample {sample_id}")
    
    def _next_sample(self):
        """Move to next sample"""
        self.current_index += 1
    
    def _previous_sample(self):
        """Move to previous sample"""
        if self.current_index > 0:
            self.current_index -= 1
    
    def _save_labeling_session(self):
        """Save labeling session statistics"""
        session_metadata = {
            'session_id': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'start_time': self.stats['session_start'],
            'end_time': datetime.now().isoformat(),
            'total_samples': self.stats['total_samples'],
            'labeled_samples': self.labeled_count,
            'digit_counts': self.stats['digit_counts'],
            'data_directory': self.data_dir
        }
        
        session_file = os.path.join(self.data_dir, f"labeling_session_{session_metadata['session_id']}.json")
        with open(session_file, 'w') as f:
            json.dump(session_metadata, f, indent=2)
        
        print(f"Labeling session saved to {session_file}")
    
    def _print_final_stats(self):
        """Print final labeling statistics"""
        print(f"\n=== Labeling Complete ===")
        print(f"Total samples: {self.stats['total_samples']}")
        print(f"Labeled samples: {self.labeled_count}")
        print(f"Labeling rate: {self.labeled_count/self.stats['total_samples']*100:.1f}%")
        print("\nDigit distribution:")
        for digit, count in self.stats['digit_counts'].items():
            if count > 0:
                print(f"  {digit}: {count} samples")
        print("=" * 30)

def main():
    parser = argparse.ArgumentParser(description='Label collected digit samples')
    parser.add_argument('--data-dir', default='../data',
                       help='Directory containing collected data')
    
    args = parser.parse_args()
    
    # Create labeling interface
    labeler = DigitLabelingInterface(data_dir=args.data_dir)
    
    # Start labeling
    labeler.start_labeling()

if __name__ == "__main__":
    main()
