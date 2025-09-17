#!/usr/bin/env python3
"""
High-Performance Threaded Digit Recognition with Camera Stream
Uses threading to separate detection from display for better FPS.
"""

import cv2
import numpy as np
import time
import os
import sys
import threading
import queue
from collections import defaultdict
from digit_detector import DigitDetector

class DigitCameraThreaded:
    def __init__(self, stream_url="http://192.168.1.200:8080/video"):
        self.stream_url = stream_url
        self.detector = DigitDetector()
        
        # Threading and synchronization
        self.detection_queue = queue.Queue(maxsize=1)  # Only keep latest frame
        self.result_queue = queue.Queue(maxsize=10)    # Store recent results
        self.detection_thread = None
        self.running = False
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.last_detection_time = 0
        self.detection_interval = 0.2  # Run detection every 200ms
        
        # Digit tracking
        self.digit_predictions = {}  # Store predictions by digit ID
        self.digit_confidence = {}   # Store confidence scores
        self.digit_positions = {}    # Store positions for tracking
        
    def connect_to_stream(self):
        """Connect to the MJPEG stream"""
        print("🎥 Connecting to camera stream...")
        print(f"📡 Stream URL: {self.stream_url}")
        
        # Connect to the stream with optimized settings
        self.cap = cv2.VideoCapture(self.stream_url)
        
        if not self.cap.isOpened():
            print("❌ Failed to connect to camera stream")
            print("💡 Make sure the Windows webcam server is running on port 8080")
            return False
        
        # Optimize stream settings for better quality
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to prevent lag
        self.cap.set(cv2.CAP_PROP_FPS, 30)        # Set target FPS
        
        # Get stream properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print("✅ Connected to camera stream!")
        print(f"📐 Stream resolution: {self.width}x{self.height}")
        print(f"🎬 Stream FPS: {self.fps}")
        
        return True
    
    def run_cuda_inference(self, digit_image):
        """Run CUDA inference on the detected digit"""
        try:
            # Check if CUDA inference executable exists
            cuda_path = "../bin/inference"
            if not os.path.exists(cuda_path):
                return None
            
            # Create a temporary file with a unique name
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as temp_file:
                temp_filename = temp_file.name
                digit_image.astype(np.float32).tofile(temp_filename)
            
            # Run the CUDA inference
            import subprocess
            result = subprocess.run([cuda_path, temp_filename], capture_output=True, text=True, timeout=3)
            
            # Clean up immediately
            os.unlink(temp_filename)
            
            if result.returncode == 0:
                # Parse the prediction result
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if line.startswith('Prediction:'):
                        prediction = line.split(':')[1].strip()
                        if prediction.isdigit():
                            return int(prediction)
            return None
                
        except Exception as e:
            return None
        finally:
            # Ensure cleanup
            try:
                if 'temp_filename' in locals() and os.path.exists(temp_filename):
                    os.unlink(temp_filename)
            except:
                pass
    
    def detection_worker(self):
        """Worker thread for digit detection and inference"""
        while self.running:
            try:
                # Get frame from queue (non-blocking)
                frame = self.detection_queue.get(timeout=0.1)
                
                # Detect digits
                digit_regions = self.detector.detect_digits(frame)
                
                if digit_regions:
                    # Process each detected digit
                    for i, (digit_region, bbox) in enumerate(digit_regions):
                        # Run inference
                        prediction = self.run_cuda_inference(digit_region)
                        
                        if prediction is not None:
                            # Create digit ID based on position (simple tracking)
                            x, y, w, h = bbox
                            digit_id = f"{x//50}_{y//50}"  # Grid-based ID
                            
                            # Store result
                            result = {
                                'prediction': prediction,
                                'bbox': bbox,
                                'timestamp': time.time(),
                                'digit_id': digit_id
                            }
                            
                            # Add to result queue (non-blocking)
                            try:
                                self.result_queue.put_nowait(result)
                            except queue.Full:
                                # Remove oldest result and add new one
                                try:
                                    self.result_queue.get_nowait()
                                    self.result_queue.put_nowait(result)
                                except queue.Empty:
                                    pass
                
                # Mark task as done
                self.detection_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"⚠️ Detection worker error: {e}")
                continue
    
    def get_digit_id_from_position(self, bbox):
        """Generate a stable digit ID based on position"""
        x, y, w, h = bbox
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Create grid-based ID for stability
        grid_size = 100
        grid_x = center_x // grid_size
        grid_y = center_y // grid_size
        
        return f"{grid_x}_{grid_y}"
    
    def update_digit_tracking(self, result):
        """Update digit tracking with new result"""
        digit_id = result['digit_id']
        current_time = time.time()
        
        # Update prediction if this is newer or more confident
        if (digit_id not in self.digit_predictions or 
            current_time - self.digit_predictions[digit_id]['timestamp'] > 1.0):  # Update every 1 second
            
            self.digit_predictions[digit_id] = {
                'prediction': result['prediction'],
                'bbox': result['bbox'],
                'timestamp': current_time
            }
    
    def draw_overlay(self, frame):
        """Draw information overlay on the frame"""
        # Add frame counter and FPS
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        
        # Create semi-transparent background for text
        overlay = frame.copy()
        
        # Draw background rectangles for better text visibility
        cv2.rectangle(overlay, (5, 5), (250, 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Add info overlay with better font settings
        cv2.putText(frame, f"FPS: {fps:.1f}", (15, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Add detection status
        detection_fps = 1.0 / self.detection_interval if self.detection_interval > 0 else 0
        cv2.putText(frame, f"Detection: {detection_fps:.1f}Hz", (15, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
        
        # Draw digit predictions
        current_time = time.time()
        for digit_id, data in self.digit_predictions.items():
            # Only show recent predictions (within last 3 seconds)
            if current_time - data['timestamp'] < 3.0:
                bbox = data['bbox']
                prediction = data['prediction']
                x, y, w, h = bbox
                
                # Draw bounding box with better visibility
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                
                # Create background for prediction text
                text_bg_x = x
                text_bg_y = y - 35
                text_bg_w = 80
                text_bg_h = 30
                
                # Draw semi-transparent background for prediction
                overlay = frame.copy()
                cv2.rectangle(overlay, (text_bg_x, text_bg_y), 
                             (text_bg_x + text_bg_w, text_bg_y + text_bg_h), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                
                # Draw prediction with better font
                cv2.putText(frame, f"Digit: {prediction}", (x + 5, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
                
                # Draw age indicator with better font
                age = current_time - data['timestamp']
                cv2.putText(frame, f"{age:.1f}s", (x + 5, y + h + 20), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
        
        return frame
    
    def run(self):
        """Main processing loop"""
        if not self.connect_to_stream():
            return
        
        # Create window
        cv2.namedWindow("Threaded Digit Recognition", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Threaded Digit Recognition", 1280, 720)
        
        print("🎬 Threaded digit recognition started. Press 'q' to quit, 's' to save frame.")
        print("💡 Write digits on paper and hold them up to the camera!")
        
        # Start detection thread
        self.running = True
        self.detection_thread = threading.Thread(target=self.detection_worker, daemon=True)
        self.detection_thread.start()
        
        while True:
            # Clear buffered frames
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
            
            # Process results from detection thread
            while not self.result_queue.empty():
                try:
                    result = self.result_queue.get_nowait()
                    self.update_digit_tracking(result)
                except queue.Empty:
                    break
            
            # Draw overlay
            display_frame = self.draw_overlay(frame)
            
            # Display frame
            cv2.imshow("Threaded Digit Recognition", display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"threaded_frame_{int(time.time())}.jpg"
                cv2.imwrite(filename, display_frame)
                print(f"📸 Frame saved as {filename}")
            elif key == ord('r'):
                # Reset tracking
                self.digit_predictions.clear()
                self.frame_count = 0
                self.start_time = time.time()
                print("🔄 Tracking reset")
            elif key == ord('+'):
                # Increase detection frequency
                self.detection_interval = max(0.05, self.detection_interval - 0.05)
                print(f"🔍 Detection interval: {self.detection_interval:.2f}s")
            elif key == ord('-'):
                # Decrease detection frequency
                self.detection_interval = min(1.0, self.detection_interval + 0.05)
                print(f"🔍 Detection interval: {self.detection_interval:.2f}s")
        
        # Cleanup
        self.running = False
        if self.detection_thread:
            self.detection_thread.join(timeout=1.0)
        
        self.cap.release()
        cv2.destroyAllWindows()
        print("👋 Threaded digit recognition stopped")

def main():
    stream_url = "http://192.168.1.200:8080/video"
    
    digit_camera = DigitCameraThreaded(stream_url)
    
    try:
        digit_camera.run()
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
