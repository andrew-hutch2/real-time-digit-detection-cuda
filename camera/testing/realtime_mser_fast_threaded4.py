#!/usr/bin/env python3
"""
Real-time MSER Digit Detection with Multi-Scale Strategy
Multi-scale detection with intelligent scale selection and parallel processing
"""

import cv2
import numpy as np
import threading
import time
import queue
from collections import deque
import concurrent.futures

class MultiScaleMSERDigitDetector:
    def __init__(self):
        # Multi-scale detection parameters
        self.scales = [0.25, 0.5, 0.75, 1.0, 1.5]  # 5 scales for comprehensive detection
        self.scale_specific_params = {
            0.25: {'delta': 2, 'min_area': 50, 'max_area': 5000, 'max_variation': 0.15, 'min_diversity': 0.1},
            0.5:  {'delta': 3, 'min_area': 75, 'max_area': 15000, 'max_variation': 0.2, 'min_diversity': 0.12},
            0.75: {'delta': 4, 'min_area': 100, 'max_area': 30000, 'max_variation': 0.25, 'min_diversity': 0.15},
            1.0:  {'delta': 5, 'min_area': 150, 'max_area': 60000, 'max_variation': 0.3, 'min_diversity': 0.18},
            1.5:  {'delta': 6, 'min_area': 200, 'max_area': 120000, 'max_variation': 0.35, 'min_diversity': 0.2}
        }
        
        # Detection parameters
        self.min_aspect_ratio = 0.15
        self.max_aspect_ratio = 2.5
        self.min_size = 15
        self.max_size = 200
        
        # Border weighting
        self.border_weight_threshold = 0.3
        
        # Enhanced preprocessing parameters
        self.clahe_clip_limit = 2.0
        self.clahe_tile_size = (8, 8)
        self.morphology_kernel_size = 3
        self.contrast_alpha = 1.2
        self.brightness_beta = 10
        
        # Multi-scale strategy parameters
        self.max_parallel_scales = 3  # Process max 3 scales in parallel
        self.scale_confidence_weights = {0.25: 0.8, 0.5: 1.0, 0.75: 1.0, 1.0: 0.9, 1.5: 0.7}
        
    def create_mser_for_scale(self, scale):
        """Create MSER detector with scale-specific parameters"""
        params = self.scale_specific_params[scale]
        return cv2.MSER_create(
            delta=params['delta'],
            min_area=params['min_area'],
            max_area=params['max_area'],
            max_variation=params['max_variation'],
            min_diversity=params['min_diversity'],
            max_evolution=200,
            area_threshold=1.01,
            min_margin=0.003,
            edge_blur_size=4
        )
    
    def enhance_contrast_clahe(self, gray):
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_tile_size
        )
        return clahe.apply(gray)
    
    def enhance_contrast_linear(self, gray):
        """Apply linear contrast enhancement"""
        return cv2.convertScaleAbs(gray, alpha=self.contrast_alpha, beta=self.brightness_beta)
    
    def apply_morphological_operations(self, thresh):
        """Apply morphological operations to clean up small objects"""
        # Create kernel for morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.morphology_kernel_size, self.morphology_kernel_size))
        
        # Apply opening to remove noise
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Apply closing to fill gaps in small objects
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        
        return closed
    
    def apply_bilateral_filter(self, gray):
        """Apply bilateral filter for noise reduction while preserving edges"""
        return cv2.bilateralFilter(gray, 9, 75, 75)
    
    def preprocess_frame_multi_scale(self, frame):
        """Enhanced preprocessing for multi-scale detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Step 1: Noise reduction with bilateral filter
        denoised = self.apply_bilateral_filter(gray)
        
        # Step 2: CLAHE for local contrast enhancement
        clahe_enhanced = self.enhance_contrast_clahe(denoised)
        
        # Step 3: Linear contrast enhancement
        contrast_enhanced = self.enhance_contrast_linear(clahe_enhanced)
        
        # Step 4: Apply Gaussian blur to reduce noise while preserving small details
        blurred = cv2.GaussianBlur(contrast_enhanced, (3, 3), 0)
        
        # Step 5: Apply adaptive thresholding for better small object contrast
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Step 6: Morphological operations to clean up small noise and fill gaps
        cleaned = self.apply_morphological_operations(thresh)
        
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
    
    def detect_at_scale(self, frame, scale):
        """Detect objects at a specific scale"""
        # Preprocess frame
        processed = self.preprocess_frame_multi_scale(frame)
        
        # Scale down for processing
        h, w = processed.shape
        new_h, new_w = int(h * scale), int(w * scale)
        scaled = cv2.resize(processed, (new_w, new_h))
        
        # Create MSER detector for this scale
        mser = self.create_mser_for_scale(scale)
        
        # Detect MSER regions
        regions, _ = mser.detectRegions(scaled)
        
        detections = []
        for region in regions:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(region)
            
            # Scale back up to original frame
            x = int(x / scale)
            y = int(y / scale)
            w = int(w / scale)
            h = int(h / scale)
            
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
            
            # Calculate confidence based on scale, region stability, and border weight
            scale_weight = self.scale_confidence_weights.get(scale, 1.0)
            confidence = weight * 0.5 + scale_weight * 0.3 + 0.2  # Combined confidence
            
            detections.append({
                'bbox': (x, y, w, h),
                'confidence': confidence,
                'weight': weight,
                'scale': scale
            })
        
        return detections
    
    def detect_multi_scale(self, frame):
        """Detect objects using multi-scale strategy with parallel processing"""
        # Select scales to process (intelligent selection)
        scales_to_process = self.select_scales_intelligently(frame)
        
        # Process scales in parallel
        all_detections = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_parallel_scales) as executor:
            # Submit detection tasks for each scale
            future_to_scale = {
                executor.submit(self.detect_at_scale, frame, scale): scale 
                for scale in scales_to_process
            }
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_scale):
                scale = future_to_scale[future]
                try:
                    detections = future.result()
                    all_detections.extend(detections)
                except Exception as e:
                    print(f"Error processing scale {scale}: {e}")
        
        # Remove duplicates using advanced NMS
        final_detections = self.remove_duplicates_advanced(all_detections)
        
        return final_detections
    
    def select_scales_intelligently(self, frame):
        """Intelligently select scales to process based on frame characteristics"""
        # For now, use all scales, but this could be enhanced to select based on:
        # - Frame size
        # - Previous detection results
        # - Performance requirements
        return self.scales
    
    def remove_duplicates_advanced(self, detections):
        """Advanced duplicate removal for multi-scale detection"""
        if not detections:
            return []
        
        # Sort by confidence (highest first)
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        filtered = []
        for det in detections:
            x1, y1, w1, h1 = det['bbox']
            is_duplicate = False
            
            for existing in filtered:
                x2, y2, w2, h2 = existing['bbox']
                
                # Calculate overlap
                overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                overlap_area = overlap_x * overlap_y
                
                area1 = w1 * h1
                area2 = w2 * h2
                union_area = area1 + area2 - overlap_area
                
                if union_area > 0:
                    overlap_ratio = overlap_area / union_area
                    
                    # Scale-aware overlap threshold
                    scale_factor = min(det['scale'], existing['scale']) / max(det['scale'], existing['scale'])
                    threshold = 0.4 * scale_factor  # More lenient for different scales
                    
                    if overlap_ratio > threshold:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                filtered.append(det)
        
        return filtered

class MultiScaleMSERThreadedDetector:
    def __init__(self, stream_url="http://192.168.1.200:8080/video"):
        self.stream_url = stream_url
        self.detector = MultiScaleMSERDigitDetector()
        
        # Threading
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=5)
        self.running = False
        
        # Detection caching
        self.latest_detections = []
        self.latest_detection_time = 0
        self.detection_timeout = 0.5  # 500ms timeout
        
        # Performance tracking
        self.detection_count = 0
        self.fps_counter = 0
        self.fps_start_time = time.time()
        
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
                
                # Perform multi-scale detection
                detections = self.detector.detect_multi_scale(frame)
                
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
                self.detection_count += result['total_count']
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
            weight = det['weight']
            scale = det['scale']
            
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
            
            # Draw scale and weight info
            scale_label = f"S: {scale:.2f} W: {weight:.2f}"
            cv2.putText(overlay, scale_label, (x, y + h + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return overlay
    
    def run(self):
        """Main detection loop"""
        self.running = True
        
        # Start threads
        capture_thread = threading.Thread(target=self.capture_frames, daemon=True)
        detection_thread = threading.Thread(target=self.detection_worker, daemon=True)
        
        capture_thread.start()
        detection_thread.start()
        
        print("Multi-Scale MSER Digit Detection Started")
        print("Press 'q' to quit, 's' to save screenshot")
        
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
                
                # Add info overlay
                self.fps_counter += 1
                if time.time() - self.fps_start_time >= 1.0:
                    fps = self.fps_counter / (time.time() - self.fps_start_time)
                    self.fps_counter = 0
                    self.fps_start_time = time.time()
                else:
                    fps = 0
                
                # Status info
                cache_status = "FRESH" if time.time() - self.latest_detection_time < 0.1 else "CACHED"
                detection_age = time.time() - self.latest_detection_time
                
                info_text = [
                    f"Multi-Scale MSER Detection",
                    f"FPS: {fps:.1f}",
                    f"Detections: {len(detections)}",
                    f"Total: {self.detection_count}",
                    f"Status: {cache_status}",
                    f"Age: {detection_age:.2f}s"
                ]
                
                y_offset = 30
                for text in info_text:
                    cv2.putText(display_frame, text, (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    y_offset += 25
                
                # Display frame
                cv2.imshow("Multi-Scale MSER Digit Detection", display_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"Multi_Scale_MSER_Detection_{timestamp}.png"
                    cv2.imwrite(filename, display_frame)
                    print(f"Screenshot saved: {filename}")
                
                self.frame_queue.task_done()
                
        except KeyboardInterrupt:
            print("\nStopping detection...")
        finally:
            self.running = False
            cv2.destroyAllWindows()
            
            # Wait for threads to finish
            capture_thread.join(timeout=2)
            detection_thread.join(timeout=2)

if __name__ == "__main__":
    detector = MultiScaleMSERThreadedDetector()
    detector.run()
