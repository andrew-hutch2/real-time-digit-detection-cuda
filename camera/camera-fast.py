#!/usr/bin/env python3
"""
Fast Digit Recognition Camera with Optimized Concurrent Processing
Simplified high-performance implementation with non-blocking inference.
"""

import cv2
import time
import subprocess
import threading
import queue
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from detectDigit_aggressive import AggressiveDigitDetectionCamera
import hashlib

class FastDigitRecognitionCamera:
    """High-performance digit recognition camera with non-blocking inference"""
    
    def __init__(self, stream_url: str = "http://192.168.1.100:8080/video"):
        self.detection_camera = AggressiveDigitDetectionCamera(stream_url)
        
        # Backend integration
        self.inference_binary = "../bin/inference"
        self.weights_file = "retrained_model_best250epoch.bin"
        
        # Prediction results - simplified
        self.predictions = {}
        self.pending_predictions = {}
        self.prediction_lock = threading.Lock()
        
        # Thread pool for inference - separate from stream
        self.inference_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="inference")
        
        # Simple caching to avoid redundant inference
        self.inference_cache = {}
        self.cache_lock = threading.Lock()
        
        # Performance settings
        self.inference_timeout = 2.0
        self.cache_hit_threshold = 0.95
        self.inference_cooldown = 0.5  # Minimum time between inferences for same digit
        
        # Tracking for stability
        self.digit_tracking = {}
        self.last_inference_time = {}
        
        # Performance monitoring
        self.stats = {
            'frames_processed': 0,
            'inferences_made': 0,
            'cache_hits': 0,
            'start_time': time.time()
        }
        
        # Reset functionality
        self.last_reset_time = 0
        self.auto_reset_enabled = True
        self.detection_quality_history = []
        self.quality_history_length = 10
    
    def create_stable_key(self, bbox):
        """Create a stable key for tracking digits"""
        x, y, w, h = bbox
        center_x, center_y = x + w//2, y + h//2
        
        # Use coarser granularity for stability
        grid_x = center_x // 30  # 30 pixel grid
        grid_y = center_y // 30
        size_bucket = (w + h) // 50  # Size bucket
        
        return f"{grid_x}_{grid_y}_{size_bucket}"
    
    def reset_detection_state(self):
        """Reset detection state to help with repositioned paper"""
        current_time = time.time()
        
        # Only allow reset every 2 seconds to prevent spam
        if current_time - self.last_reset_time < 2.0:
            return
        
        self.last_reset_time = current_time
        
        with self.prediction_lock:
            # Clear all predictions
            self.predictions.clear()
            self.pending_predictions.clear()
            
            # Clear tracking data
            self.digit_tracking.clear()
            self.last_inference_time.clear()
            
            # Clear cache
            with self.cache_lock:
                self.inference_cache.clear()
        
        print("🔄 Detection state reset - ready for repositioned paper!")
    
    def check_auto_reset(self, detections):
        """Check if automatic reset should be triggered based on detection quality"""
        if not self.auto_reset_enabled:
            return
        
        # Calculate detection quality metrics
        detection_count = len(detections)
        avg_confidence = np.mean([det['confidence'] for det in detections]) if detections else 0
        
        # Store quality metrics
        quality_score = detection_count * avg_confidence
        self.detection_quality_history.append(quality_score)
        
        # Keep only recent history
        if len(self.detection_quality_history) > self.quality_history_length:
            self.detection_quality_history.pop(0)
        
        # Check if quality has dropped significantly
        if len(self.detection_quality_history) >= 5:
            recent_avg = np.mean(self.detection_quality_history[-3:])  # Last 3 frames
            older_avg = np.mean(self.detection_quality_history[:-3])   # Earlier frames
            
            # If recent quality is much lower than older quality, trigger reset
            if older_avg > 0 and recent_avg < older_avg * 0.3:  # 70% drop
                print("🔄 Auto-reset triggered due to detection quality drop")
                self.reset_detection_state()
                self.detection_quality_history.clear()  # Clear history after reset
    
    def get_digit_hash(self, digit_array):
        """Create a hash of the digit data for caching"""
        return hashlib.md5(digit_array.tobytes()).hexdigest()
    
    def check_cache(self, digit_array):
        """Check if we have a cached result for this digit"""
        digit_hash = self.get_digit_hash(digit_array)
        with self.cache_lock:
            if digit_hash in self.inference_cache:
                return self.inference_cache[digit_hash]
        return None
    
    def cache_result(self, digit_array, prediction):
        """Cache the inference result"""
        digit_hash = self.get_digit_hash(digit_array)
        with self.cache_lock:
            self.inference_cache[digit_hash] = prediction
            # Limit cache size
            if len(self.inference_cache) > 200:
                # Remove oldest entry
                oldest_key = next(iter(self.inference_cache))
                del self.inference_cache[oldest_key]
    
    def run_inference(self, digit_array, temp_file_path):
        """Run CUDA inference on a digit - non-blocking"""
        try:
            # Save digit to binary file
            if not self.detection_camera.detector.save_digit_binary(digit_array, temp_file_path):
                return None
            
            # Run CUDA inference
            cmd = [self.inference_binary, temp_file_path, self.weights_file]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.inference_timeout, cwd="../bin")
            
            if result.returncode == 0:
                # Parse prediction
                stdout_lines = result.stdout.strip().split('\n')
                prediction_line = [line for line in stdout_lines if line.startswith('Prediction:')]
                if prediction_line:
                    prediction = prediction_line[0].split(':')[1].strip()
                    try:
                        digit_prediction = int(prediction)
                        return {'digit': digit_prediction, 'confidence': 1.0}
                    except ValueError:
                        return None
            return None
                
        except (subprocess.TimeoutExpired, Exception):
            return None
    
    def _run_inference_async(self, stable_key, digit_array, bbox, center):
        """Run inference asynchronously - this runs in separate thread"""
        temp_file_path = None
        try:
            # Check cache first
            cached_prediction = self.check_cache(digit_array)
            if cached_prediction:
                prediction = cached_prediction
                self.stats['cache_hits'] += 1
            else:
                # Get unique temp file
                temp_file_path = f"../bin/temp_digit_{stable_key}_{int(time.time()*1000)}.bin"
                
                # Run inference
                prediction = self.run_inference(digit_array, temp_file_path)
                
                # Cache result if successful
                if prediction:
                    self.cache_result(digit_array, prediction)
                    self.stats['inferences_made'] += 1
            
            # Update predictions
            with self.prediction_lock:
                # Remove from pending
                if stable_key in self.pending_predictions:
                    del self.pending_predictions[stable_key]
                
                if prediction:
                    self.predictions[stable_key] = {
                        'prediction': prediction,
                        'bbox': bbox,
                        'center': center,
                        'timestamp': time.time(),
                        'last_seen': time.time()
                    }
                    
        except Exception as e:
            print(f"Inference error for {stable_key}: {e}")
            # Clean up on error
            with self.prediction_lock:
                if stable_key in self.pending_predictions:
                    del self.pending_predictions[stable_key]
        finally:
            # Clean up temp file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except Exception:
                    pass
    
    def cleanup_old_predictions(self):
        """Clean up old predictions"""
        current_time = time.time()
        
        with self.prediction_lock:
            # Clean up old predictions
            keys_to_remove = []
            for key, pred_data in self.predictions.items():
                if current_time - pred_data['last_seen'] > 5.0:  # 5 second timeout
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.predictions[key]
            
            # Clean up old inference times
            old_times = []
            for key, last_time in self.last_inference_time.items():
                if current_time - last_time > 10.0:
                    old_times.append(key)
            
            for key in old_times:
                del self.last_inference_time[key]
    
    def prediction_worker(self):
        """Worker thread for running predictions - separate from stream"""
        while self.detection_camera.running:
            try:
                # Periodic cleanup
                current_time = time.time()
                if current_time % 2.0 < 0.1:  # Every 2 seconds
                    self.cleanup_old_predictions()
                
                detections = self.detection_camera.get_latest_detections()
                
                for det in detections:
                    if det['confidence'] > 0.4:  # Lower threshold for faster detection
                        stable_key = self.create_stable_key(det['bbox'])
                        x, y, w, h = det['bbox']
                        center_x, center_y = x + w//2, y + h//2
                        
                        # Check if we should process this digit
                        should_process = False
                        
                        with self.prediction_lock:
                            if stable_key not in self.predictions:
                                if stable_key not in self.pending_predictions:
                                    # Check cooldown
                                    last_time = self.last_inference_time.get(stable_key, 0)
                                    if current_time - last_time > self.inference_cooldown:
                                        should_process = True
                        
                        if should_process:
                            # Mark as pending
                            with self.prediction_lock:
                                self.pending_predictions[stable_key] = {
                                    'timestamp': current_time,
                                    'bbox': det['bbox']
                                }
                            
                            # Submit to thread pool - non-blocking
                            self.inference_executor.submit(
                                self._run_inference_async,
                                stable_key,
                                det['preprocessed_digit'],
                                det['bbox'],
                                (center_x, center_y)
                            )
                            
                            # Record inference time
                            self.last_inference_time[stable_key] = current_time
                
                time.sleep(0.05)  # Small sleep to prevent busy waiting
                
            except Exception as e:
                print(f"Prediction worker error: {e}")
                time.sleep(0.1)
    
    def draw_prediction_overlay(self, frame, detections):
        """Draw detection results with predictions - clean and simple"""
        overlay = frame.copy()
        current_time = time.time()
        
        # Update stats
        self.stats['frames_processed'] += 1
        
        # Update last_seen for current detections
        with self.prediction_lock:
            for det in detections:
                if det['confidence'] > 0.6:
                    stable_key = self.create_stable_key(det['bbox'])
                    if stable_key in self.predictions:
                        self.predictions[stable_key]['last_seen'] = current_time
        
        # Draw detections and predictions
        for det in detections:
            if det['confidence'] > 0.4:
                x, y, w, h = det['bbox']
                stable_key = self.create_stable_key(det['bbox'])
                
                # Draw bounding box - simple green
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Check for prediction
                with self.prediction_lock:
                    if stable_key in self.predictions:
                        pred_data = self.predictions[stable_key]
                        digit = pred_data['prediction']['digit']
                        
                        # Draw prediction below the box - simple black text
                        cv2.putText(overlay, str(digit), (x, y + h + 25), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        
        return overlay
    
    def start_recognition(self):
        """Start the fast digit recognition system"""
        print("🚀 Starting Fast Digit Recognition Camera")
        print("Controls:")
        print("  'q' - Quit")
        print("  'f' - Toggle fullscreen")
        print("  'r' - Reset detection state")
        print("Features: Non-blocking inference, Simple caching, Clean display")
        
        # Start detection camera
        self.detection_camera.running = True
        
        # Start threads - inference runs separately from stream
        capture_thread = threading.Thread(target=self.detection_camera.capture_frames, daemon=True)
        detection_thread = threading.Thread(target=self.detection_camera.detection_worker, daemon=True)
        prediction_thread = threading.Thread(target=self.prediction_worker, daemon=True)
        
        capture_thread.start()
        detection_thread.start()
        prediction_thread.start()
        
        # Create window in fullscreen
        cv2.namedWindow("Fast Digit Recognition", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Fast Digit Recognition", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        is_fullscreen = True
        
        try:
            while True:
                try:
                    # Get frame from stream - this is the main loop, no blocking
                    frame = self.detection_camera.frame_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Get detections
                detections = self.detection_camera.get_latest_detections()
                
                # Check for automatic reset
                self.check_auto_reset(detections)
                
                # Draw overlay - this is fast, no inference here
                display_frame = self.draw_prediction_overlay(frame, detections)
                
                # Show frame
                cv2.imshow("Fast Digit Recognition", display_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('f'):
                    # Toggle fullscreen
                    is_fullscreen = not is_fullscreen
                    if is_fullscreen:
                        cv2.setWindowProperty("Fast Digit Recognition", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    else:
                        cv2.setWindowProperty("Fast Digit Recognition", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                        cv2.resizeWindow("Fast Digit Recognition", 1280, 720)
                elif key == ord('r'):
                    # Reset detection state
                    self.reset_detection_state()
                
                self.detection_camera.frame_queue.task_done()
                
        except KeyboardInterrupt:
            pass
        finally:
            self.detection_camera.running = False
            cv2.destroyAllWindows()
            
            # Shutdown thread pool
            self.inference_executor.shutdown(wait=True)
            
            # Wait for threads to finish
            capture_thread.join(timeout=2)
            detection_thread.join(timeout=2)
            prediction_thread.join(timeout=2)
            
            # Print final stats
            self._print_final_stats()
    
    def _print_final_stats(self):
        """Print final performance statistics"""
        elapsed = time.time() - self.stats['start_time']
        
        print("\n" + "="*40)
        print("FINAL PERFORMANCE STATISTICS")
        print("="*40)
        print(f"Total runtime: {elapsed:.2f} seconds")
        print(f"Frames processed: {self.stats['frames_processed']}")
        print(f"Inferences made: {self.stats['inferences_made']}")
        print(f"Cache hits: {self.stats['cache_hits']}")
        print(f"Average FPS: {self.stats['frames_processed'] / elapsed:.2f}")
        print(f"Cache hit rate: {self.stats['cache_hits'] / max(1, self.stats['inferences_made']):.1%}")
        print("="*40)

def main():
    camera = FastDigitRecognitionCamera()
    camera.start_recognition()

if __name__ == "__main__":
    main()