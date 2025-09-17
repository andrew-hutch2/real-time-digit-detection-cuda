#!/usr/bin/env python3
"""
Real-time Digit Recognition with Camera Stream
Integrates MJPEG camera stream with digit detection and CUDA inference.
"""

import cv2
import numpy as np
import time
import os
import sys
from digit_detector import DigitDetector

class DigitCamera:
    def __init__(self, stream_url="http://192.168.1.200:8080/video"):
        self.stream_url = stream_url
        self.detector = DigitDetector()
        self.frame_count = 0
        self.start_time = time.time()
        self.last_detection_time = 0
        self.detection_cooldown = 0.5  # Minimum time between detections (seconds)
        
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
                print("⚠️ CUDA inference executable not found. Run 'cd .. && make inference' first.")
                return None
            
            # Create a temporary file with a unique name to avoid conflicts
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as temp_file:
                temp_filename = temp_file.name
                # Write the preprocessed digit data
                digit_image.astype(np.float32).tofile(temp_filename)
            
            # Run the CUDA inference
            import subprocess
            result = subprocess.run([cuda_path, temp_filename], capture_output=True, text=True, timeout=5)
            
            # Clean up temporary file immediately
            os.unlink(temp_filename)
            
            if result.returncode == 0:
                # Parse the prediction result - look for "Prediction: X" line
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if line.startswith('Prediction:'):
                        prediction = line.split(':')[1].strip()
                        if prediction.isdigit():
                            return int(prediction)
                
                print(f"⚠️ Could not parse prediction from: {result.stdout}")
                return None
            else:
                print(f"⚠️ CUDA inference failed: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            print("⚠️ CUDA inference timed out")
            return None
        except Exception as e:
            print(f"⚠️ CUDA inference error: {e}")
            return None
        finally:
            # Ensure temp file is cleaned up even if there's an error
            try:
                if 'temp_filename' in locals() and os.path.exists(temp_filename):
                    os.unlink(temp_filename)
            except:
                pass
    
    def process_frame(self, frame):
        """Process a single frame for digit detection and recognition"""
        current_time = time.time()
        
        # Check if enough time has passed since last detection
        if current_time - self.last_detection_time < self.detection_cooldown:
            return frame, None
        
        # Detect digits in the frame
        digit_regions = self.detector.detect_digits(frame)
        
        if not digit_regions:
            return frame, None
        
        # Process the first detected digit
        digit_region, bbox = digit_regions[0]
        
        # Run CUDA inference
        prediction = self.run_cuda_inference(digit_region)
        
        if prediction is not None:
            self.last_detection_time = current_time
            return frame, (prediction, bbox)
        
        return frame, None
    
    def draw_overlay(self, frame, prediction_info=None):
        """Draw information overlay on the frame"""
        # Add frame counter and FPS
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        
        # Add info overlay
        #cv2.putText(frame, f"Frame: {self.frame_count}", (10, 30), 
                   #cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        #cv2.putText(frame, f"Resolution: {self.width}x{self.height}", (10, 90), 
                   #cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add digit recognition result
        if prediction_info:
            prediction, bbox = prediction_info
            x, y, w, h = bbox
            
            # Draw bounding box around detected digit
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw prediction result
            cv2.putText(frame, f"Digit: {prediction}", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            
            # Add confidence indicator (since we don't have actual confidence)
            cv2.putText(frame, "CUDA Model", (x, y + h + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        return frame
    
    def run(self):
        """Main processing loop"""
        if not self.connect_to_stream():
            return
        
        # Create window with high-quality settings
        cv2.namedWindow("Digit Recognition", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Digit Recognition", 1280, 720)
        
        # Set window properties for better display
        cv2.setWindowProperty("Digit Recognition", cv2.WND_PROP_TOPMOST, 0)
        
        print("🎬 Digit recognition started. Press 'q' to quit, 's' to save frame.")
        
        while True:
            # Clear any buffered frames to prevent lag
            for _ in range(2):
                self.cap.grab()
            
            ret, frame = self.cap.read()
            
            if not ret:
                print("⚠️ Failed to read frame")
                break
            
            # Process frame for digit detection
            processed_frame, prediction_info = self.process_frame(frame)
            
            # Draw overlay with results
            display_frame = self.draw_overlay(processed_frame, prediction_info)
            
            # Display frame
            cv2.imshow("Digit Recognition", display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                filename = f"digit_frame_{int(time.time())}.jpg"
                cv2.imwrite(filename, display_frame)
                print(f"📸 Frame saved as {filename}")
            elif key == ord('r'):
                # Reset frame counter
                self.frame_count = 0
                self.start_time = time.time()
                print("🔄 Frame counter reset")
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        print("👋 Digit recognition stopped")

def main():
    # You can change the IP address here if needed
    stream_url = "http://192.168.1.200:8080/video"
    
    # Create and run the digit camera
    digit_camera = DigitCamera(stream_url)
    
    try:
        digit_camera.run()
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
