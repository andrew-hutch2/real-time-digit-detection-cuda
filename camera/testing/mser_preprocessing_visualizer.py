#!/usr/bin/env python3
"""
MSER-based Preprocessing Visualizer for Digit Detection

This script captures frames from the webcam and saves preprocessed images
at different stages of the MSER-based pipeline so you can visually inspect how
MSER detection affects digit detection quality compared to traditional methods.

Key differences from traditional approach:
- Uses MSER (Maximally Stable Extremal Regions) instead of contour detection
- Implements border weighting to devalue screen edges
- Shows step-by-step MSER processing stages
- More robust to noise and paper edges

Usage:
    python3 mser_preprocessing_visualizer.py [stream_url]
    
Controls:
    's' - Save current frame and all MSER preprocessing stages
    'q' - Quit
    'r' - Reset counter
"""

import cv2
import numpy as np
import os
import sys
import time
from datetime import datetime

# Add parent directory to path to import MSER detector
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from mser_digit_detector import MSERDigitDetector

class MSERPreprocessingVisualizer:
    def __init__(self, stream_url="http://192.168.1.200:8080/video"):
        self.stream_url = stream_url
        self.detector = MSERDigitDetector()
        self.frame_count = 0
        self.saved_count = 0
        
        # No output directory needed - screenshots saved to current directory
        
    def connect_to_stream(self):
        """Connect to the MJPEG stream"""
        print("🎥 Connecting to camera stream...")
        print(f"📡 Stream URL: {self.stream_url}")
        
        self.cap = cv2.VideoCapture(self.stream_url)
        
        if not self.cap.isOpened():
            print("❌ Failed to connect to camera stream")
            print("💡 Make sure the Windows webcam server is running on port 8080")
            return False
        
        # Optimize stream settings
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Get stream properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print("✅ Connected to camera stream!")
        print(f"📐 Stream resolution: {self.width}x{self.height}")
        print(f"🎬 Stream FPS: {self.fps}")
        
        return True
    
    def save_mser_preprocessing_stages(self, frame):
        """Save all MSER preprocessing stages for visual inspection"""
        self.saved_count += 1
        print(f"💾 Saving MSER preprocessing stages #{self.saved_count}...")
        
        # 1. Original frame
        original_path = f"mser_{self.saved_count:03d}_01_original.jpg"
        cv2.imwrite(original_path, frame)
        
        # 2. Grayscale conversion
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        gray_path = f"mser_{self.saved_count:03d}_02_grayscale.jpg"
        cv2.imwrite(gray_path, gray)
        
        # 3. Gaussian blur (MSER preprocessing)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        blur_path = f"mser_{self.saved_count:03d}_03_blurred.jpg"
        cv2.imwrite(blur_path, blurred)
        
        # 4. Border weight mask visualization
        height, width = gray.shape
        border_mask = self.detector.create_border_weight_mask(height, width)
        # Convert to 0-255 range for visualization
        border_vis = (border_mask * 255).astype(np.uint8)
        border_path = f"mser_{self.saved_count:03d}_04_border_weight_mask.jpg"
        cv2.imwrite(border_path, border_vis)
        
        # 5. MSER regions detection (visualization)
        mser_regions = self.detector.detect_mser_regions(blurred, border_mask)
        
        # Create visualization of all MSER regions
        mser_vis = gray.copy()
        for i, (mask, rect, weight_score) in enumerate(mser_regions):
            x, y, w, h = rect
            # Draw region outline
            cv2.rectangle(mser_vis, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # Add weight score
            cv2.putText(mser_vis, f"{weight_score:.2f}", (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        mser_path = f"mser_{self.saved_count:03d}_05_mser_regions.jpg"
        cv2.imwrite(mser_path, mser_vis)
        
        # 6. Filtered MSER regions (only those that passed filtering)
        filtered_vis = gray.copy()
        filtered_count = 0
        for i, (mask, rect, weight_score) in enumerate(mser_regions):
            if weight_score > 0.2:  # Same threshold as in detector
                x, y, w, h = rect
                cv2.rectangle(filtered_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(filtered_vis, f"MSER {filtered_count}", (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                filtered_count += 1
        
        filtered_path = f"mser_{self.saved_count:03d}_06_filtered_regions.jpg"
        cv2.imwrite(filtered_path, filtered_vis)
        
        # 7. Detect digits and save individual digit regions
        digit_regions = self.detector.detect_digits(frame)
        
        if digit_regions:
            print(f"   Found {len(digit_regions)} potential digits using MSER")
            
            for i, (preprocessed_digit, bbox) in enumerate(digit_regions):
                x, y, w, h = bbox
                
                # Save original digit region (before preprocessing)
                digit_roi = self.detector.extract_digit_region(frame, bbox)
                roi_path = f"mser_{self.saved_count:03d}_07_digit_{i}_roi.jpg"
                cv2.imwrite(roi_path, digit_roi)
                
                # Save preprocessed digit (28x28)
                # Convert back to 0-255 range for saving
                digit_for_save = (preprocessed_digit * 255).astype(np.uint8)
                preprocessed_path = f"mser_{self.saved_count:03d}_08_digit_{i}_preprocessed.jpg"
                cv2.imwrite(preprocessed_path, digit_for_save)
                
                # Save preprocessed digit with contrast enhancement for better visibility
                enhanced = cv2.equalizeHist(digit_for_save)
                enhanced_path = f"mser_{self.saved_count:03d}_09_digit_{i}_enhanced.jpg"
                cv2.imwrite(enhanced_path, enhanced)
                
                print(f"   MSER Digit {i}: bbox=({x},{y},{w},{h})")
        else:
            print("   No digits detected with MSER")
        
        # 8. Save frame with MSER detection overlay
        vis_frame = frame.copy()
        for i, (_, bbox) in enumerate(digit_regions):
            x, y, w, h = bbox
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(vis_frame, f"MSER {i}", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        overlay_path = f"mser_{self.saved_count:03d}_10_detection_overlay.jpg"
        cv2.imwrite(overlay_path, vis_frame)
        
        print(f"✅ Saved {len(digit_regions)} MSER digit regions to {self.output_dir}")
    
    def draw_overlay(self, frame):
        """Draw information overlay on the frame"""
        self.frame_count += 1
        
        # Add info overlay
        cv2.putText(frame, f"Frame: {self.frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Saved: {self.saved_count}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "MSER Detection - Press 's' to save, 'q' to quit", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def run(self):
        """Main processing loop"""
        if not self.connect_to_stream():
            return
        
        # Create window
        cv2.namedWindow("MSER Preprocessing Visualizer", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("MSER Preprocessing Visualizer", 1280, 720)
        
        print("🎬 MSER preprocessing visualizer started.")
        print("📋 Controls:")
        print("   's' - Save current frame and all MSER preprocessing stages")
        print("   'q' - Quit")
        print("   'r' - Reset counter")
        print("")
        print("🔍 MSER Detection Features:")
        print("   - Maximally Stable Extremal Regions for robust digit detection")
        print("   - Border weighting to reduce false detections from screen edges")
        print("   - Step-by-step visualization of the detection process")
        print("")
        
        while True:
            # Clear buffered frames
            for _ in range(2):
                self.cap.grab()
            
            ret, frame = self.cap.read()
            
            if not ret:
                print("⚠️ Failed to read frame")
                break
            
            # Draw overlay
            display_frame = self.draw_overlay(frame)
            
            # Display frame
            cv2.imshow("MSER Preprocessing Visualizer", display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_mser_preprocessing_stages(frame)
            elif key == ord('r'):
                self.frame_count = 0
                self.saved_count = 0
                print("🔄 Counters reset")
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        print(f"👋 MSER preprocessing visualizer stopped. Check {self.output_dir} for saved images.")

def main():
    # Get stream URL from command line or use default
    stream_url = sys.argv[1] if len(sys.argv) > 1 else "http://192.168.1.200:8080/video"
    
    # Create and run the visualizer
    visualizer = MSERPreprocessingVisualizer(stream_url)
    
    try:
        visualizer.run()
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
