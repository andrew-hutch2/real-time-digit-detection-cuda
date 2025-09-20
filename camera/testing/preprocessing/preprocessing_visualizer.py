#!/usr/bin/env python3
"""
Preprocessing Visualizer for Digit Recognition
Captures and visualizes preprocessed digits before inference to debug preprocessing issues.
"""

import cv2
import numpy as np
import time
import os
import threading
import queue
from datetime import datetime
import sys
# Add the camera directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from detectDigit import DigitDetectionCamera

class PreprocessingVisualizer:
    def __init__(self, stream_url="http://192.168.1.100:8080/video"):
        self.detection_camera = DigitDetectionCamera(stream_url)
        
        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"preprocessing_output_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Capture settings
        self.max_captures = 10  # Maximum digits to capture per run
        self.captured_count = 0
        self.capture_interval = 2.0  # Seconds between captures
        self.last_capture_time = 0
        
        print(f"📁 Output directory: {self.output_dir}")
        print(f"🎯 Will capture up to {self.max_captures} digits")
        print("Press 'c' to capture current detections, 'q' to quit")
    
    def save_preprocessed_digit(self, digit_roi, preprocessed_digit, bbox, confidence, capture_id):
        """Save digit in multiple formats for analysis"""
        
        # 1. Save original ROI
        roi_filename = f"{self.output_dir}/digit_{capture_id:02d}_original.png"
        cv2.imwrite(roi_filename, digit_roi)
        
        # 2. Save preprocessed digit (28x28) - what the model actually receives
        # Convert from MNIST normalized values back to [0,255] for visualization
        # First denormalize: (normalized * std) + mean
        mean, std = 0.1307, 0.3081
        denormalized = (preprocessed_digit * std) + mean
        # Clip to [0,1] range and convert to [0,255]
        denormalized = np.clip(denormalized, 0, 1)
        preprocessed_vis = (denormalized * 255).astype(np.uint8)
        preprocessed_filename = f"{self.output_dir}/digit_{capture_id:02d}_preprocessed_28x28.png"
        cv2.imwrite(preprocessed_filename, preprocessed_vis)
        
        # 2b. Save binary version (pure black/white, no gray)
        # Ensure binary values: 0 for background, 255 for digits
        binary_vis = np.where(preprocessed_digit > 0.5, 255, 0).astype(np.uint8)
        binary_filename = f"{self.output_dir}/digit_{capture_id:02d}_binary_28x28.png"
        cv2.imwrite(binary_filename, binary_vis)
        
        # 2c. Save debug info about the preprocessing
        debug_info = f"{self.output_dir}/digit_{capture_id:02d}_debug.txt"
        with open(debug_info, 'w') as f:
            f.write(f"Preprocessing Debug Info for Digit {capture_id}\n")
            f.write(f"==========================================\n\n")
            f.write(f"Original ROI shape: {digit_roi.shape}\n")
            f.write(f"Preprocessed shape: {preprocessed_digit.shape}\n")
            f.write(f"Preprocessed min: {preprocessed_digit.min():.4f}\n")
            f.write(f"Preprocessed max: {preprocessed_digit.max():.4f}\n")
            f.write(f"Preprocessed mean: {preprocessed_digit.mean():.4f}\n")
            f.write(f"Preprocessed std: {preprocessed_digit.std():.4f}\n")
            f.write(f"MNIST normalization: mean=0.1307, std=0.3081\n")
            f.write(f"Gaussian blur: kernel=(3,3), sigma=0.5\n")
            f.write(f"Brightness enhancement: 1.3x multiplier\n\n")
            
            # Count pixel values
            unique_vals, counts = np.unique(preprocessed_digit, return_counts=True)
            f.write(f"Unique values and counts:\n")
            for val, count in zip(unique_vals, counts):
                f.write(f"  {val:.4f}: {count} pixels\n")
            
            # Check if it's properly inverted
            zero_pixels = np.sum(preprocessed_digit == 0.0)
            one_pixels = np.sum(preprocessed_digit == 1.0)
            f.write(f"\nBinary analysis:\n")
            f.write(f"  Black pixels (0.0): {zero_pixels}\n")
            f.write(f"  White pixels (1.0): {one_pixels}\n")
            f.write(f"  Background ratio: {zero_pixels/(zero_pixels+one_pixels):.2%}\n")
            f.write(f"  Digit ratio: {one_pixels/(zero_pixels+one_pixels):.2%}\n")
            
            if zero_pixels > one_pixels:
                f.write(f"  ✅ Correct: Black background with white digits (MNIST format)\n")
            else:
                f.write(f"  ❌ Incorrect: White background with black digits\n")
        
        # 3. Save enlarged version for better visibility
        enlarged = cv2.resize(preprocessed_vis, (280, 280), interpolation=cv2.INTER_NEAREST)
        enlarged_filename = f"{self.output_dir}/digit_{capture_id:02d}_preprocessed_280x280.png"
        cv2.imwrite(enlarged_filename, enlarged)
        
        # 4. Save comparison image (original + preprocessed side by side)
        comparison = self.create_comparison_image(digit_roi, preprocessed_vis, bbox, confidence)
        comparison_filename = f"{self.output_dir}/digit_{capture_id:02d}_comparison.png"
        cv2.imwrite(comparison_filename, comparison)
        
        # 4b. Save binary comparison (original + binary version)
        binary_comparison = self.create_comparison_image(digit_roi, binary_vis, bbox, confidence, "Binary (Pure B/W)")
        binary_comparison_filename = f"{self.output_dir}/digit_{capture_id:02d}_binary_comparison.png"
        cv2.imwrite(binary_comparison_filename, binary_comparison)
        
        # 5. Save raw data for analysis
        data_filename = f"{self.output_dir}/digit_{capture_id:02d}_data.txt"
        with open(data_filename, 'w') as f:
            f.write(f"Capture ID: {capture_id}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Bounding Box: {bbox}\n")
            f.write(f"Confidence: {confidence:.3f}\n")
            f.write(f"Original ROI Shape: {digit_roi.shape}\n")
            f.write(f"Preprocessed Shape: {preprocessed_digit.shape}\n")
            f.write(f"Preprocessed Min: {preprocessed_digit.min():.4f}\n")
            f.write(f"Preprocessed Max: {preprocessed_digit.max():.4f}\n")
            f.write(f"Preprocessed Mean: {preprocessed_digit.mean():.4f}\n")
            f.write(f"Preprocessed Std: {preprocessed_digit.std():.4f}\n")
        
        print(f"✅ Captured digit {capture_id}: {roi_filename}")
        return True
    
    def create_comparison_image(self, original_roi, preprocessed_vis, bbox, confidence, label="Preprocessed"):
        """Create side-by-side comparison of original and preprocessed digit"""
        # Resize original ROI to match preprocessed size for comparison
        original_resized = cv2.resize(original_roi, (28, 28))
        
        # Create comparison image
        comparison = np.zeros((28, 60), dtype=np.uint8)  # 28x60 to fit both images
        comparison[:, :28] = original_resized
        comparison[:, 32:] = preprocessed_vis
        
        # Add separator line
        comparison[:, 28:32] = 128
        
        # Enlarge for better visibility
        comparison_large = cv2.resize(comparison, (600, 280), interpolation=cv2.INTER_NEAREST)
        
        # Add text labels
        cv2.putText(comparison_large, "Original", (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
        cv2.putText(comparison_large, label, (320, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
        cv2.putText(comparison_large, f"Conf: {confidence:.2f}", (10, 260), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255, 1)
        
        return comparison_large
    
    def capture_current_detections(self):
        """Capture all current detections"""
        current_time = time.time()
        
        # Rate limiting
        if current_time - self.last_capture_time < self.capture_interval:
            print(f"⏳ Please wait {self.capture_interval - (current_time - self.last_capture_time):.1f}s before next capture")
            return
        
        if self.captured_count >= self.max_captures:
            print(f"📊 Reached maximum captures ({self.max_captures})")
            return
        
        detections = self.detection_camera.get_latest_detections()
        
        if not detections:
            print("❌ No detections found")
            return
        
        print(f"📸 Found {len(detections)} detections:")
        for i, det in enumerate(detections):
            print(f"   Detection {i+1}: confidence={det['confidence']:.3f}, bbox={det['bbox']}")
        
        high_conf_count = 0
        for i, det in enumerate(detections):
            if self.captured_count >= self.max_captures:
                break
                
            if det['confidence'] > 0.5:  # Only capture high-confidence detections
                high_conf_count += 1
                self.captured_count += 1
                self.save_preprocessed_digit(
                    det['digit_roi'],
                    det['preprocessed_digit'],
                    det['bbox'],
                    det['confidence'],
                    self.captured_count
                )
        
        if high_conf_count == 0:
            print(f"⚠️  No high-confidence detections (confidence > 0.5) found")
            print(f"   Try lowering the confidence threshold or check detection quality")
        
        self.last_capture_time = current_time
        print(f"📊 Total captured: {self.captured_count}/{self.max_captures}")
    
    def run_visualizer(self):
        """Run the preprocessing visualizer"""
        print("🎥 Starting Preprocessing Visualizer")
        print("Controls:")
        print("  'c' - Capture current detections")
        print("  'q' - Quit")
        print("  's' - Show preprocessing stats")
        print("  'd' - Show detection debug info")
        
        # Start detection camera
        self.detection_camera.running = True
        
        # Start threads
        capture_thread = threading.Thread(target=self.detection_camera.capture_frames, daemon=True)
        detection_thread = threading.Thread(target=self.detection_camera.detection_worker, daemon=True)
        
        capture_thread.start()
        detection_thread.start()
        
        # Create window
        cv2.namedWindow("Preprocessing Visualizer", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Preprocessing Visualizer", 1280, 720)
        
        try:
            while True:
                try:
                    frame = self.detection_camera.frame_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                detections = self.detection_camera.get_latest_detections()
                display_frame = self.draw_visualizer_overlay(frame, detections)
                
                cv2.imshow("Preprocessing Visualizer", display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    self.capture_current_detections()
                elif key == ord('s'):
                    self.show_preprocessing_stats()
                elif key == ord('d'):
                    self.show_detection_debug()
                
                self.detection_camera.frame_queue.task_done()
                
        except KeyboardInterrupt:
            pass
        finally:
            self.detection_camera.running = False
            cv2.destroyAllWindows()
            
            # Wait for threads to finish
            capture_thread.join(timeout=2)
            detection_thread.join(timeout=2)
            
            print(f"\n📁 All captures saved to: {self.output_dir}")
            print(f"📊 Total digits captured: {self.captured_count}")
    
    def draw_visualizer_overlay(self, frame, detections):
        """Draw overlay with capture information"""
        overlay = frame.copy()
        
        # Add capture info
        cv2.putText(overlay, f"Captured: {self.captured_count}/{self.max_captures}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw detections
        for i, det in enumerate(detections):
            if det['confidence'] > 0.5:
                x, y, w, h = det['bbox']
                
                # Color based on confidence
                if det['confidence'] > 0.8:
                    color = (0, 255, 0)  # Green - high confidence
                elif det['confidence'] > 0.6:
                    color = (0, 255, 255)  # Yellow - medium confidence
                else:
                    color = (0, 165, 255)  # Orange - low confidence
                
                # Draw bounding box
                cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
                
                # Draw label
                label = f"Digit {i+1}: {det['confidence']:.2f}"
                cv2.putText(overlay, label, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Add instructions
        cv2.putText(overlay, "Press 'c' to capture, 'q' to quit", 
                   (10, overlay.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return overlay
    
    def show_preprocessing_stats(self):
        """Show statistics about captured preprocessing"""
        if self.captured_count == 0:
            print("❌ No digits captured yet")
            return
        
        print(f"\n📊 Preprocessing Statistics:")
        print(f"   Total captured: {self.captured_count}")
        print(f"   Output directory: {self.output_dir}")
        print(f"   Files per digit: 7 (original, preprocessed, binary, debug, enlarged, comparison, binary_comparison)")
        print(f"   Total files: {self.captured_count * 7}")
    
    def show_detection_debug(self):
        """Show current detection debug information"""
        detections = self.detection_camera.get_latest_detections()
        
        print(f"\n🔍 Detection Debug Info:")
        print(f"   Total detections: {len(detections)}")
        
        if detections:
            confidences = [det['confidence'] for det in detections]
            print(f"   Confidence range: {min(confidences):.3f} - {max(confidences):.3f}")
            print(f"   Average confidence: {sum(confidences)/len(confidences):.3f}")
            print(f"   High confidence (>0.5): {sum(1 for c in confidences if c > 0.5)}")
            print(f"   Medium confidence (0.3-0.5): {sum(1 for c in confidences if 0.3 <= c <= 0.5)}")
            print(f"   Low confidence (<0.3): {sum(1 for c in confidences if c < 0.3)}")
            
            print(f"\n   Individual detections:")
            for i, det in enumerate(detections):
                x, y, w, h = det['bbox']
                print(f"     {i+1}: conf={det['confidence']:.3f}, bbox=({x},{y},{w},{h})")
        else:
            print("   No detections found - check camera stream and detection parameters")

def main():
    visualizer = PreprocessingVisualizer()
    visualizer.run_visualizer()

if __name__ == "__main__":
    main()
