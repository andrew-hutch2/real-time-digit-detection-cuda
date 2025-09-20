#!/usr/bin/env python3
"""
High-Performance Fast Threaded MSER-based Digit Detection

This script provides ultra-fast real-time MSER digit detection using threading
and multiple performance optimizations for maximum speed while maintaining
good detection quality.

Performance optimizations:
- Frame downscaling for detection (4x speed improvement)
- Simplified MSER parameters for speed
- Reduced border weighting complexity
- Fast duplicate removal
- Minimal preprocessing steps
- Threaded detection for smooth display

Usage:
    python3 realtime_mser_fast_threaded.py [stream_url]
    
Controls:
    'q' - Quit
    'r' - Reset detection counter
    's' - Save current frame with detections
    '+' - Increase detection frequency
    '-' - Decrease detection frequency
"""

import cv2
import numpy as np
import os
import sys
import time
import threading
import queue
from datetime import datetime

# Add parent directory to path to import fast MSER detector
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from mser_digit_detection_fast import FastMSERDigitDetector

class FastThreadedMSERDetection:
    def __init__(self, stream_url="http://192.168.1.100:8080/video"):
        self.stream_url = stream_url
        self.detector = FastMSERDigitDetector()
        
        # Threading and synchronization
        self.detection_queue = queue.Queue(maxsize=1)  # Only keep latest frame
        self.result_queue = queue.Queue(maxsize=10)    # Store recent results
        self.detection_thread = None
        self.running = False
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.last_detection_time = 0
        self.detection_interval = 0.1  # Run detection every 100ms (10Hz) for fast detector
        self.detection_count = 0
        self.detection_times = []  # Track detection performance
        
        # Detection caching
        self.latest_detections = []  # Cache the latest complete detection
        self.latest_detection_time = 0
        self.detection_timeout = 2.0  # Keep detections for 2 seconds (shorter for fast updates)
        
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
        
        # Optimize stream settings for maximum performance
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer
        self.cap.set(cv2.CAP_PROP_FPS, 30)        # Target FPS
        
        # Get stream properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print("✅ Connected to camera stream!")
        print(f"📐 Stream resolution: {self.width}x{self.height}")
        print(f"🎬 Stream FPS: {self.fps}")
        
        return True
    
    def detection_worker(self):
        """Worker thread for fast MSER digit detection"""
        while self.running:
            try:
                # Get frame from queue (non-blocking)
                frame = self.detection_queue.get(timeout=0.1)
                
                # Time the detection
                detection_start = time.time()
                
                # Run fast MSER detection
                digit_regions = self.detector.detect_digits(frame)
                
                detection_time = time.time() - detection_start
                self.detection_times.append(detection_time)
                
                # Keep only recent detection times (last 10)
                if len(self.detection_times) > 10:
                    self.detection_times.pop(0)
                
                # Create complete detection result
                detection_result = {
                    'detections': [],
                    'timestamp': time.time(),
                    'total_count': len(digit_regions),
                    'detection_time': detection_time
                }
                
                if digit_regions:
                    # Process each detected digit
                    for i, (preprocessed_digit, bbox) in enumerate(digit_regions):
                        detection_result['detections'].append({
                            'bbox': bbox,
                            'digit_index': i,
                            'preprocessed_digit': preprocessed_digit
                        })
                
                # Add complete detection result to queue (non-blocking)
                try:
                    self.result_queue.put_nowait(detection_result)
                except queue.Full:
                    # Remove oldest result and add new one
                    try:
                        self.result_queue.get_nowait()
                        self.result_queue.put_nowait(detection_result)
                    except queue.Empty:
                        pass
                
                # Mark task as done
                self.detection_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"⚠️ Fast MSER detection worker error: {e}")
                continue
    
    def get_latest_detections(self):
        """Get the latest cached detection results"""
        current_time = time.time()
        
        # Process all available results from queue
        while not self.result_queue.empty():
            try:
                detection_result = self.result_queue.get_nowait()
                # Update cached detections with latest complete result
                self.latest_detections = detection_result['detections']
                self.latest_detection_time = detection_result['timestamp']
                self.detection_count += detection_result['total_count']
            except queue.Empty:
                break
        
        # Return cached detections if they're still valid (within timeout)
        if current_time - self.latest_detection_time < self.detection_timeout:
            return self.latest_detections
        else:
            # Clear old detections
            self.latest_detections = []
            return []
    
    def draw_detection_overlay(self, frame, detections):
        """Draw bounding boxes and information overlay"""
        # Draw bounding boxes for detected digits
        for detection in detections:
            bbox = detection['bbox']
            x, y, w, h = bbox
            digit_index = detection['digit_index']
            
            # Draw bounding box
            color = (0, 255, 0)  # Green for detected digits
            thickness = 2
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
            
            # Draw label
            label = f"Fast {digit_index + 1}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Draw label background
            cv2.rectangle(frame, (x, y - label_size[1] - 10), 
                         (x + label_size[0], y), color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Create information overlay
        overlay_height = 120
        overlay = np.zeros((overlay_height, frame.shape[1], 3), dtype=np.uint8)
        overlay[:] = (0, 0, 0)  # Black background
        
        # Calculate FPS
        elapsed_time = time.time() - self.start_time
        fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        
        # Calculate average detection time
        avg_detection_time = np.mean(self.detection_times) if self.detection_times else 0
        
        # Calculate detection age for display
        current_time = time.time()
        detection_age = current_time - self.latest_detection_time if self.latest_detection_time > 0 else 0
        cache_status = "FRESH" if detection_age < 0.2 else "CACHED"
        
        # Add text information
        info_texts = [
            f"Fast Threaded MSER Detection - FPS: {fps:.1f}",
            f"Current: {len(detections)} digits | Total: {self.detection_count} | Status: {cache_status}",
            f"Detection: {1.0/self.detection_interval:.1f}Hz | Avg Time: {avg_detection_time*1000:.1f}ms | Age: {detection_age:.1f}s",
            f"Controls: 'q'=quit, 'r'=reset, 's'=save, '+/-'=freq"
        ]
        
        for i, text in enumerate(info_texts):
            y_pos = 25 + i * 25
            cv2.putText(overlay, text, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Combine overlay with frame
        combined = np.vstack([overlay, frame])
        
        return combined
    
    def save_current_frame(self, frame, detections):
        """Save current frame with detections"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save frame with detections
        detection_frame = self.draw_detection_overlay(frame, detections)
        save_path = f"fast_mser_detection_{timestamp}.jpg"
        cv2.imwrite(save_path, detection_frame)
        
        print(f"💾 Saved detection frame: {save_path}")
        print(f"   Detected {len(detections)} digits")
    
    def run(self):
        """Main processing loop"""
        if not self.connect_to_stream():
            return
        
        # Create window
        cv2.namedWindow("Fast Threaded MSER Digit Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Fast Threaded MSER Digit Detection", 1280, 800)
        
        print("🎬 Fast threaded MSER detection started.")
        print("📋 Controls:")
        print("   'q' - Quit")
        print("   'r' - Reset detection counter")
        print("   's' - Save current frame with detections")
        print("   '+' - Increase detection frequency")
        print("   '-' - Decrease detection frequency")
        print("")
        print("🚀 Fast MSER Detection Features:")
        print("   - Frame downscaling for 4x speed improvement")
        print("   - Simplified MSER parameters for speed")
        print("   - Fast border weighting and duplicate removal")
        print("   - Threaded detection for smooth display")
        print("   - High-frequency detection (10Hz default)")
        print("")
        
        # Start detection thread
        self.running = True
        self.detection_thread = threading.Thread(target=self.detection_worker, daemon=True)
        self.detection_thread.start()
        
        while True:
            # Clear buffered frames for real-time performance
            for _ in range(2):
                self.cap.grab()
            
            ret, frame = self.cap.read()
            
            if not ret:
                print("⚠️ Failed to read frame")
                break
            
            # Check if it's time for detection
            current_time = time.time()
            if current_time - self.last_detection_time >= self.detection_interval:
                # Add frame to detection queue (non-blocking)
                try:
                    self.detection_queue.put_nowait(frame.copy())
                    self.last_detection_time = current_time
                except queue.Full:
                    # Skip this frame if queue is full
                    pass
            
            # Get latest cached detections
            detections = self.get_latest_detections()
            
            # Draw detection overlay
            display_frame = self.draw_detection_overlay(frame, detections)
            
            # Display frame
            cv2.imshow("Fast Threaded MSER Digit Detection", display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.detection_count = 0
                self.frame_count = 0
                self.start_time = time.time()
                self.detection_times = []
                print("🔄 Detection counter reset")
            elif key == ord('s'):
                self.save_current_frame(frame, detections)
            elif key == ord('+'):
                # Increase detection frequency
                self.detection_interval = max(0.05, self.detection_interval - 0.05)
                print(f"🔍 Detection interval: {self.detection_interval:.2f}s ({1.0/self.detection_interval:.1f}Hz)")
            elif key == ord('-'):
                # Decrease detection frequency
                self.detection_interval = min(0.5, self.detection_interval + 0.05)
                print(f"🔍 Detection interval: {self.detection_interval:.2f}s ({1.0/self.detection_interval:.1f}Hz)")
            
            self.frame_count += 1
        
        # Cleanup
        self.running = False
        if self.detection_thread:
            self.detection_thread.join(timeout=1.0)
        
        self.cap.release()
        cv2.destroyAllWindows()
        print(f"👋 Fast threaded MSER detection stopped.")
        print(f"📊 Total frames processed: {self.frame_count}")
        print(f"📊 Total detections: {self.detection_count}")
        if self.detection_times:
            print(f"📊 Average detection time: {np.mean(self.detection_times)*1000:.1f}ms")
        print(f"📁 Saved frames in: {self.output_dir}")

def main():
    # Get stream URL from command line or use default
    stream_url = sys.argv[1] if len(sys.argv) > 1 else "http://192.168.1.100:8080/video"
    
    # Create and run the fast threaded detector
    detector = FastThreadedMSERDetection(stream_url)
    
    try:
        detector.run()
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
