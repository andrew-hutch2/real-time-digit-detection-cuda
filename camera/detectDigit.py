#!/usr/bin/env python3
"""
MSER Digit Detection - Clean Implementation
Focused on detection and drawing bounding boxes around digits
"""

import cv2
import numpy as np
import threading
import time
import queue

class MSERDigitDetector:
    def __init__(self):
        # MSER parameters optimized for small objects
        self.mser = cv2.MSER_create(
            delta=3,           # Lower for better small object detection
            min_area=80,       # Reduced for tiny digits
            max_area=120000,   # Increased for large digits
            max_variation=0.2, # Lower for better stability
            min_diversity=0.12, # Lower for more sensitive detection
            max_evolution=200,
            area_threshold=1.01,
            min_margin=0.003,
            edge_blur_size=4   # Reduced for sharper edges
        )
        
        # Detection parameters
        self.detection_scale = 0.5
        self.min_aspect_ratio = 0.15
        self.max_aspect_ratio = 2.5
        self.min_size = 15
        self.max_size = 200
        
        # Border weighting
        self.border_weight_threshold = 0.3
        
    def preprocess_frame(self, frame):
        """Preprocessing optimized for small object detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise while preserving small details
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Apply adaptive thresholding for better small object contrast
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Light morphological operations to clean up small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def calculate_border_weight(self, x, y, w, h, frame_shape):
        """Calculate weight based on distance from borders"""
        frame_h, frame_w = frame_shape[:2]
        
        # Calculate distances from borders
        dist_top = y
        dist_bottom = frame_h - (y + h)
        dist_left = x
        dist_right = frame_w - (x + w)
        
        # Calculate minimum distance to any border
        min_dist = min(dist_top, dist_bottom, dist_left, dist_right)
        
        # Calculate weight (higher for regions away from borders)
        max_dist = min(frame_w, frame_h) / 2
        weight = min(1.0, min_dist / max_dist)
        
        return weight
    
    def extract_digit_region(self, frame, bbox):
        """Extract the digit region from the frame with padding"""
        x, y, w, h = bbox
        
        # Add some padding
        padding = 5
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(frame.shape[1] - x, w + 2 * padding)
        h = min(frame.shape[0] - y, h + 2 * padding)
        
        # Extract region
        digit_roi = frame[y:y+h, x:x+w]
        
        return digit_roi
    
    def preprocess_digit(self, digit_roi):
        """Preprocess a digit region to 28x28 format for the model"""
        # Convert to grayscale if needed
        if len(digit_roi.shape) == 3:
            gray = cv2.cvtColor(digit_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = digit_roi.copy()
        
        # Resize to 28x28
        resized = cv2.resize(gray, (28, 28))
        
        # Invert colors (MNIST has white digits on black background)
        inverted = cv2.bitwise_not(resized)
        
        # Normalize to [0, 1] range
        normalized = inverted.astype(np.float32) / 255.0
        
        return normalized
    
    def save_digit_binary(self, digit_array, filename):
        """Save preprocessed digit to binary format for CUDA inference"""
        try:
            # Flatten to 784 dimensions
            flattened = digit_array.flatten()
            
            # Apply MNIST normalization (same as training)
            mean, std = 0.1307, 0.3081
            normalized = (flattened - mean) / std
            
            # Save as binary file
            normalized.astype(np.float32).tofile(filename)
            return True
        except Exception as e:
            print(f"Error saving digit: {e}")
            return False
    
    def detect_digits(self, frame):
        """Detect digits in frame and return with preprocessed data"""
        # Preprocess frame
        processed = self.preprocess_frame(frame)
        
        # Scale down for faster processing
        h, w = processed.shape
        new_h, new_w = int(h * self.detection_scale), int(w * self.detection_scale)
        scaled = cv2.resize(processed, (new_w, new_h))
        
        # Detect MSER regions
        regions, _ = self.mser.detectRegions(scaled)
        
        detections = []
        for region in regions:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(region)
            
            # Scale back up
            x = int(x / self.detection_scale)
            y = int(y / self.detection_scale)
            w = int(w / self.detection_scale)
            h = int(h / self.detection_scale)
            
            # Filter by size
            if w < self.min_size or h < self.min_size or w > self.max_size or h > self.max_size:
                continue
                
            # Filter by aspect ratio
            aspect_ratio = w / h
            if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                continue
            
            # Calculate border weight
            weight = self.calculate_border_weight(x, y, w, h, frame.shape)
            if weight < self.border_weight_threshold:
                continue
            
            # Calculate confidence based on region stability and border weight
            confidence = weight * 0.7 + 0.3  # Base confidence with weight influence
            
            # Extract and preprocess digit for ML inference
            digit_roi = self.extract_digit_region(frame, (x, y, w, h))
            preprocessed_digit = self.preprocess_digit(digit_roi)
            
            detections.append({
                'bbox': (x, y, w, h),
                'confidence': confidence,
                'weight': weight,
                'digit_roi': digit_roi,
                'preprocessed_digit': preprocessed_digit
            })
        
        # Remove duplicates using improved NMS
        detections = self.remove_duplicates(detections)
        
        return detections
    
    def remove_duplicates(self, detections):
        """Remove duplicate detections with nested detection handling"""
        if not detections:
            return []
        
        # Sort by area (largest first) to prioritize larger detections
        detections.sort(key=lambda x: x['bbox'][2] * x['bbox'][3], reverse=True)
        
        filtered = []
        for det in detections:
            x1, y1, w1, h1 = det['bbox']
            area1 = w1 * h1
            is_duplicate = False
            
            for existing in filtered:
                x2, y2, w2, h2 = existing['bbox']
                area2 = w2 * h2
                
                # Check if current detection is completely contained within existing
                if (x1 >= x2 and y1 >= y2 and 
                    x1 + w1 <= x2 + w2 and y1 + h1 <= y2 + h2):
                    # Current detection is inside existing one - skip it
                    is_duplicate = True
                    break
                
                # Check if existing detection is completely contained within current
                if (x2 >= x1 and y2 >= y1 and 
                    x2 + w2 <= x1 + w1 and y2 + h2 <= y1 + h1):
                    # Existing detection is inside current one - remove existing
                    filtered.remove(existing)
                    break
                
                # Calculate overlap for partial overlaps
                overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                overlap_area = overlap_x * overlap_y
                
                if overlap_area > 0:
                    union_area = area1 + area2 - overlap_area
                    overlap_ratio = overlap_area / union_area
                    
                    # If significant overlap, keep the larger detection
                    if overlap_ratio > 0.3:  # Reduced threshold for better filtering
                        if area1 > area2:
                            # Current is larger, remove existing
                            filtered.remove(existing)
                        else:
                            # Existing is larger, skip current
                            is_duplicate = True
                            break
            
            if not is_duplicate:
                filtered.append(det)
        
        return filtered

class DigitDetectionCamera:
    def __init__(self, stream_url="http://192.168.1.200:8080/video"):
        self.stream_url = stream_url
        self.detector = MSERDigitDetector()
        
        # Threading
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=5)
        self.running = False
        
        # Detection caching
        self.latest_detections = []
        self.latest_detection_time = 0
        self.detection_timeout = 0.5  # 500ms timeout
        
    def capture_frames(self):
        """Capture frames from camera stream"""
        cap = cv2.VideoCapture(self.stream_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        while self.running:
            ret, frame = cap.read()
            if ret:
                try:
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    # Remove old frame if queue is full
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(frame)
                    except queue.Empty:
                        pass
            else:
                print("Failed to read frame")
                time.sleep(0.1)
        
        cap.release()
    
    def detection_worker(self):
        """Worker thread for detection processing"""
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1.0)
                
                # Perform detection
                detections = self.detector.detect_digits(frame)
                
                # Put result in queue
                detection_result = {
                    'detections': detections,
                    'timestamp': time.time(),
                    'total_count': len(detections)
                }
                
                try:
                    self.result_queue.put_nowait(detection_result)
                except queue.Full:
                    # Remove old result if queue is full
                    try:
                        self.result_queue.get_nowait()
                        self.result_queue.put_nowait(detection_result)
                    except queue.Empty:
                        pass
                
                self.frame_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Detection error: {e}")
                time.sleep(0.1)
    
    def get_latest_detections(self):
        """Get latest detection results with caching"""
        current_time = time.time()
        
        # Process all available results
        while not self.result_queue.empty():
            try:
                result = self.result_queue.get_nowait()
                self.latest_detections = result['detections']
                self.latest_detection_time = result['timestamp']
            except queue.Empty:
                break
        
        # Return cached detections if within timeout
        if current_time - self.latest_detection_time < self.detection_timeout:
            return self.latest_detections
        
        return []
    
    def draw_detection_overlay(self, frame, detections):
        """Draw detection results on frame"""
        overlay = frame.copy()
        
        for i, det in enumerate(detections):
            x, y, w, h = det['bbox']
            confidence = det['confidence']
            
            # Only draw if confidence is above 0.5
            if confidence < 0.5:
                continue
            
            # Color based on confidence
            if confidence > 0.8:
                color = (0, 255, 0)  # Green - high confidence
            elif confidence > 0.6:
                color = (0, 255, 255)  # Yellow - medium confidence
            else:
                color = (0, 165, 255)  # Orange - low confidence
            
            # Draw bounding box
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            label = f"Digit {i+1}: {confidence:.2f}"
            cv2.putText(overlay, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return overlay
    
    def start_detection(self):
        """Start the detection system"""
        self.running = True
        
        # Start threads
        capture_thread = threading.Thread(target=self.capture_frames, daemon=True)
        detection_thread = threading.Thread(target=self.detection_worker, daemon=True)
        
        capture_thread.start()
        detection_thread.start()
        
        print("MSER Digit Detection Started")
        print("Press 'q' to quit")
        
        try:
            while True:
                # Get latest frame
                try:
                    frame = self.frame_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Get latest detections
                detections = self.get_latest_detections()
                
                # Draw overlay
                display_frame = self.draw_detection_overlay(frame, detections)
                
                # Display frame
                cv2.imshow("Digit Detection", display_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                
                self.frame_queue.task_done()
                
        except KeyboardInterrupt:
            print("\nStopping detection...")
        finally:
            self.running = False
            cv2.destroyAllWindows()
            
            # Wait for threads to finish
            capture_thread.join(timeout=2)
            detection_thread.join(timeout=2)
    
    def get_detection_boxes(self):
        """Get current detection bounding boxes for external use"""
        return self.get_latest_detections()

if __name__ == "__main__":
    camera = DigitDetectionCamera()
    camera.start_detection()
