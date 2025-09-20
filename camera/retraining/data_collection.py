#!/usr/bin/env python3
"""
Data Collection Script for Camera-Based Digit Recognition
Captures preprocessed digit samples from camera for model retraining
"""

import cv2
import numpy as np
import os
import time
import json
import argparse
from datetime import datetime
import sys
sys.path.append('..')
from detectDigit import MSERDigitDetector
import threading
import queue

class DataCollector:
    def __init__(self, output_dir="../collected_data", camera_url="http://192.168.1.100:8080/video"):
        self.output_dir = output_dir
        self.camera_url = camera_url
        
        # Create output directories
        self.raw_dir = os.path.join(output_dir, "raw_samples")
        self.preprocessed_dir = os.path.join(output_dir, "preprocessed")
        self.metadata_dir = os.path.join(output_dir, "metadata")
        
        for dir_path in [self.raw_dir, self.preprocessed_dir, self.metadata_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Initialize detector
        self.detector = MSERDigitDetector()
        
        # Collection state
        self.collection_active = False
        self.sample_count = 0
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Quality filters
        self.min_confidence = 0.05  # Minimum detection confidence
        self.max_samples_per_digit = 1000  # Limit samples per digit
        self.digit_counts = {i: 0 for i in range(10)}
        
        # Collection statistics
        self.stats = {
            'total_detections': 0,
            'accepted_samples': 0,
            'rejected_samples': 0,
            'session_start': time.time(),
            'digits_collected': {i: 0 for i in range(10)}
        }
        
        print(f"Data collection initialized. Session ID: {self.session_id}")
        print(f"Output directory: {self.output_dir}")
        print(f"Camera URL: {self.camera_url}")
    
    def start_collection(self, target_samples=100, auto_save=True):
        """Start collecting digit samples from camera"""
        print(f"\n=== Starting Data Collection ===")
        print(f"Target samples: {target_samples}")
        print(f"Auto-save: {auto_save}")
        print("Controls:")
        print("  SPACE - Save ALL digits currently detected on screen")
        print("  'q' - Quit collection")
        print("  's' - Save session metadata")
        print("  'r' - Reset counters")
        print("=" * 40)
        
        # Initialize camera
        cap = cv2.VideoCapture(self.camera_url)
        if not cap.isOpened():
            print(f"Error: Could not connect to camera at {self.camera_url}")
            return False
        
        self.collection_active = True
        frame_count = 0
        
        try:
            while self.collection_active and self.sample_count < target_samples:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read frame from camera")
                    continue
                
                frame_count += 1
                
                # Detect digits in frame
                detections = self.detector.detect_digits(frame)
                
                # Process each detection (but don't auto-save)
                for i, detection in enumerate(detections):
                    self.stats['total_detections'] += 1
                    
                    # Extract data from detection dictionary
                    preprocessed_digit = detection['preprocessed_digit']
                    bbox = detection['bbox']
                    
                    # Quality check
                    if self._quality_check(preprocessed_digit, bbox):
                        # Show detection but don't auto-save
                        print(f"Detection #{i+1}: Good quality digit detected (press SPACE to save, any other key to skip)")
                    else:
                        self.stats['rejected_samples'] += 1
                
                # Display frame with detections
                display_frame = self._draw_detections(frame, detections)
                self._display_stats(display_frame)
                
                cv2.imshow('Data Collection - Press q to quit', display_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):  # SPACE key to save ALL current detections
                    if detections:
                        saved_count = 0
                        for i, detection in enumerate(detections):
                            preprocessed_digit = detection['preprocessed_digit']
                            bbox = detection['bbox']
                            if self._quality_check(preprocessed_digit, bbox):
                                sample_id = f"{self.session_id}_{self.sample_count:06d}"
                                self._save_sample(sample_id, preprocessed_digit, bbox, frame)
                                self.sample_count += 1
                                self.stats['accepted_samples'] += 1
                                saved_count += 1
                                print(f"Sample {self.sample_count}: Saved {sample_id}")
                        if saved_count > 0:
                            print(f"Saved {saved_count} digits from current frame")
                        else:
                            print("No quality digits found in current detections")
                    else:
                        print("No digits detected in current frame")
                elif key == ord('s'):
                    self._save_session_metadata()
                    print("Session metadata saved")
                elif key == ord('r'):
                    self._reset_counters()
                    print("Counters reset")
                
                # Auto-save every 50 samples
                if auto_save and self.sample_count % 50 == 0 and self.sample_count > 0:
                    self._save_session_metadata()
                    print(f"Auto-saved at {self.sample_count} samples")
        
        except KeyboardInterrupt:
            print("\nCollection interrupted by user")
        
        finally:
            self.collection_active = False
            cap.release()
            cv2.destroyAllWindows()
            
            # Final save
            self._save_session_metadata()
            self._print_final_stats()
    
    def _quality_check(self, digit_array, bbox):
        """Check if digit sample meets quality criteria"""
        # Check if we've hit the limit for any digit
        if max(self.digit_counts.values()) >= self.max_samples_per_digit:
            return False
        
        # Check digit array properties
        if digit_array is None or digit_array.size == 0:
            return False
        
        # Check for reasonable pixel distribution
        mean_val = np.mean(digit_array)
        std_val = np.std(digit_array)
        
        # Reject if too uniform (likely noise)
        if std_val < 0.1:
            return False
        
        # Check bounding box size
        x, y, w, h = bbox
        if w < 5 or h < 5 or w > 200 or h > 200:
            return False
        
        return True
    
    def _save_sample(self, sample_id, preprocessed_digit, bbox, original_frame):
        """Save a digit sample with metadata"""
        # Save preprocessed digit (28x28 normalized)
        digit_file = os.path.join(self.preprocessed_dir, f"{sample_id}.bin")
        preprocessed_digit.flatten().astype(np.float32).tofile(digit_file)
        
        # Save raw digit region from original frame
        x, y, w, h = bbox
        raw_digit = original_frame[y:y+h, x:x+w]
        raw_file = os.path.join(self.raw_dir, f"{sample_id}.jpg")
        cv2.imwrite(raw_file, raw_digit)
        
        # Save metadata
        metadata = {
            'sample_id': sample_id,
            'timestamp': time.time(),
            'bbox': bbox,
            'preprocessed_shape': preprocessed_digit.shape,
            'digit_file': os.path.relpath(digit_file, self.output_dir),
            'raw_file': os.path.relpath(raw_file, self.output_dir),
            'label': None,  # To be filled during labeling
            'confidence': None  # To be filled during labeling
        }
        
        metadata_file = os.path.join(self.metadata_dir, f"{sample_id}.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _draw_detections(self, frame, detections):
        """Draw detection boxes on frame"""
        display_frame = frame.copy()
        
        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            x, y, w, h = bbox
            
            # Draw bounding box
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw detection number
            cv2.putText(display_frame, f"#{i+1}", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return display_frame
    
    def _display_stats(self, frame):
        """Display collection statistics on frame"""
        stats_text = [
            f"Samples: {self.sample_count}",
            f"Detections: {self.stats['total_detections']}",
            f"Accepted: {self.stats['accepted_samples']}",
            f"Rejected: {self.stats['rejected_samples']}"
        ]
        
        y_offset = 30
        for text in stats_text:
            cv2.putText(frame, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 25
    
    def _save_session_metadata(self):
        """Save session metadata and statistics"""
        session_metadata = {
            'session_id': self.session_id,
            'start_time': self.stats['session_start'],
            'end_time': time.time(),
            'duration': time.time() - self.stats['session_start'],
            'total_samples': self.sample_count,
            'statistics': self.stats,
            'output_directory': self.output_dir,
            'camera_url': self.camera_url
        }
        
        metadata_file = os.path.join(self.output_dir, f"session_{self.session_id}.json")
        with open(metadata_file, 'w') as f:
            json.dump(session_metadata, f, indent=2)
    
    def _reset_counters(self):
        """Reset collection counters"""
        self.sample_count = 0
        self.stats['accepted_samples'] = 0
        self.stats['rejected_samples'] = 0
        self.stats['total_detections'] = 0
        self.digit_counts = {i: 0 for i in range(10)}
        print("Counters reset")
    
    def _print_final_stats(self):
        """Print final collection statistics"""
        duration = time.time() - self.stats['session_start']
        print(f"\n=== Collection Complete ===")
        print(f"Session ID: {self.session_id}")
        print(f"Duration: {duration:.1f} seconds")
        print(f"Total samples collected: {self.sample_count}")
        print(f"Total detections: {self.stats['total_detections']}")
        print(f"Acceptance rate: {self.stats['accepted_samples']/(self.stats['total_detections']+1e-6)*100:.1f}%")
        print(f"Output directory: {self.output_dir}")
        print("=" * 30)

def main():
    parser = argparse.ArgumentParser(description='Collect digit samples from camera for retraining')
    parser.add_argument('--output-dir', default='../collected_data', 
                       help='Output directory for collected data')
    parser.add_argument('--camera-url', default='http://192.168.1.100:8080/video',
                       help='Camera stream URL')
    parser.add_argument('--target-samples', type=int, default=100,
                       help='Target number of samples to collect')
    parser.add_argument('--no-auto-save', action='store_true',
                       help='Disable auto-save every 50 samples')
    
    args = parser.parse_args()
    
    # Create data collector
    collector = DataCollector(
        output_dir=args.output_dir,
        camera_url=args.camera_url
    )
    
    # Start collection
    collector.start_collection(
        target_samples=args.target_samples,
        auto_save=not args.no_auto_save
    )

if __name__ == "__main__":
    main()
