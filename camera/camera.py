#!/usr/bin/env python3
"""
Digit Recognition Camera with Backend Integration
Connects to MJPEG stream, detects digits, and predicts them using CUDA backend.
"""

import cv2
import time
import subprocess
import threading
import queue
import os
from concurrent.futures import ThreadPoolExecutor
from detectDigit import DigitDetectionCamera

class DigitRecognitionCamera:
    def __init__(self, stream_url="http://192.168.1.100:8080/video"):
        self.detection_camera = DigitDetectionCamera(stream_url)
        
        # Backend integration - weights file is now in bin directory with inference binary
        self.inference_binary = "../bin/inference"
        self.temp_digit_file = "../bin/temp_digit_0.bin"
        self.temp_file_counter = 0  # For generating unique temp files
        
        # Prediction results
        self.latest_predictions = {}
        self.pending_predictions = {}  # Track predictions in progress
        self.detection_to_prediction = {}  # Map detection to prediction result
        self.prediction_lock = threading.Lock()  # Thread safety for predictions
        
        # Thread pool for inference (allow multiple concurrent inferences with unique temp files)
        self.inference_executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="inference")
        
        # Timeout and cleanup settings
        self.inference_timeout = 3.0  # Reduced timeout for faster failure detection
        self.prediction_cleanup_interval = 2.0  # Clean up old predictions every 2 seconds
        self.last_cleanup_time = time.time()
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        
        # Matching configuration
        self.use_fallback_matching = True  # Set to False for strict one-to-one mapping
        
        # Track active futures to prevent thread pool deadlock
        self.active_futures = set()
        self.temp_file_lock = threading.Lock()  # Thread safety for temp file counter
        
        # Inference caching to avoid re-processing similar digits
        self.inference_cache = {}  # Hash of digit data -> prediction result
        self.cache_hit_threshold = 0.95  # Similarity threshold for cache hits
        
        # Throttling to reduce unnecessary inference calls
        self.last_inference_time = {}  # Track last inference time per stable_key
        self.inference_cooldown = 1.0  # Minimum time between inferences for same digit (seconds)
    
    def create_stable_key(self, bbox):
        """Create a more stable key for tracking digits"""
        x, y, w, h = bbox
        center_x, center_y = x + w//2, y + h//2
        
        # Use coarser granularity to reduce flickering
        # This makes the key more stable for moving digits
        grid_x = center_x // 20  # 20 pixel grid
        grid_y = center_y // 20  # 20 pixel grid
        size_bucket = (w + h) // 40  # Size bucket (roughly 40 pixel increments)
        
        return f"{grid_x}_{grid_y}_{size_bucket}"
    
    def iou(self, boxA, boxB):
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
        
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]
        
        return interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    
    def find_best_matching_prediction(self, bbox):
        """Find the best matching prediction for a given bounding box with strict matching rules"""
        x, y, w, h = bbox
        center_x, center_y = x + w//2, y + h//2
        
        best_match = None
        best_score = 0.0  # Use score instead of distance for better matching
        
        # Don't acquire lock here - caller should already have it
        for key, pred_data in self.detection_to_prediction.items():
            pred_center = pred_data['center']
            pred_bbox = pred_data['bbox']
            pred_w, pred_h = pred_bbox[2], pred_bbox[3]
            
            # Calculate distance between centers
            distance = ((center_x - pred_center[0])**2 + (center_y - pred_center[1])**2)**0.5
            
            # Stricter distance threshold - scale with digit size
            max_distance = min(40, 0.5 * (w + h))
            
            # Check size similarity - reject if width/height differ too much
            width_diff = abs(w - pred_w)
            height_diff = abs(h - pred_h)
            max_width_diff = 0.3 * w
            max_height_diff = 0.3 * h
            
            # Calculate IoU overlap
            iou_score = self.iou(bbox, pred_bbox)
            
            # Only consider matches that meet all criteria:
            # 1. Distance is within threshold
            # 2. Size is similar enough
            # 3. IoU overlap is significant (> 20%)
            if (distance < max_distance and 
                width_diff < max_width_diff and 
                height_diff < max_height_diff and
                iou_score > 0.2):
                
                # Calculate combined score (IoU weighted more heavily)
                combined_score = (iou_score * 0.7) + ((1.0 - distance/max_distance) * 0.3)
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_match = key
        
        return best_match
    
    def get_digit_hash(self, digit_array):
        """Create a hash of the digit data for caching"""
        import hashlib
        # Convert to bytes and hash
        digit_bytes = digit_array.tobytes()
        return hashlib.md5(digit_bytes).hexdigest()
    
    def check_inference_cache(self, digit_array):
        """Check if we have a cached result for this digit"""
        digit_hash = self.get_digit_hash(digit_array)
        if digit_hash in self.inference_cache:
            return self.inference_cache[digit_hash]
        return None
    
    def cache_inference_result(self, digit_array, prediction):
        """Cache the inference result"""
        digit_hash = self.get_digit_hash(digit_array)
        self.inference_cache[digit_hash] = prediction
        
        # Limit cache size to prevent memory bloat
        if len(self.inference_cache) > 100:
            # Remove oldest entries (simple FIFO)
            oldest_key = next(iter(self.inference_cache))
            del self.inference_cache[oldest_key]
    
    def get_unique_temp_file(self):
        """Generate a unique temporary file path for each inference"""
        with self.temp_file_lock:
            self.temp_file_counter += 1
            return f"../bin/temp_digit_{self.temp_file_counter}.bin"
    
    def cleanup_stuck_futures(self):
        """Clean up stuck futures to prevent thread pool deadlock"""
        current_time = time.time()
        stuck_futures = []
        
        for future in list(self.active_futures):
            if future.done():
                # Future completed, remove it
                self.active_futures.discard(future)
            else:
                # Check if future is stuck (running too long)
                # This is a simple check - in practice you'd want to track start time
                if len(self.active_futures) > 0:  # If we have active futures
                    # For now, just limit the number of concurrent futures
                    if len(self.active_futures) >= 3:  # Max 3 concurrent inferences
                        stuck_futures.append(future)
        
        # Cancel stuck futures
        for future in stuck_futures:
            future.cancel()
            self.active_futures.discard(future)
        
        if stuck_futures:
            print(f"Cancelled {len(stuck_futures)} stuck inference tasks")
        
    def run_cuda_inference(self, digit_array, temp_file_path):
        """Run CUDA inference on a preprocessed digit"""
        try:
            # Save digit to binary file
            if not self.detection_camera.detector.save_digit_binary(digit_array, temp_file_path):
                return None
            
            # Run CUDA inference from the bin directory (where weights file is located)
            cmd = [self.inference_binary, temp_file_path, "retrained_model_best.bin"]
            
            # Change to the bin directory where weights file is located
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.inference_timeout, cwd="../bin")
            
            if result.returncode == 0:
                # Parse the prediction from stdout (look for "Prediction: X")
                stdout_lines = result.stdout.strip().split('\n')
                prediction_line = [line for line in stdout_lines if line.startswith('Prediction:')]
                if prediction_line:
                    prediction = prediction_line[0].split(':')[1].strip()
                    try:
                        digit_prediction = int(prediction)
                        return {'digit': digit_prediction, 'confidence': 1.0}
                    except ValueError:
                        return None
                else:
                    return None
            else:
                return None
                
        except (subprocess.TimeoutExpired, Exception):
            return None
    
    def _run_inference_async(self, stable_key, digit_array, bbox, center):
        """Run inference asynchronously and update results"""
        temp_file_path = None
        try:
            # Check cache first
            cached_prediction = self.check_inference_cache(digit_array)
            if cached_prediction:
                prediction = cached_prediction
            else:
                # Get unique temp file for this inference
                temp_file_path = self.get_unique_temp_file()
                
                # Run inference with unique temp file
                prediction = self.run_cuda_inference(digit_array, temp_file_path)
                
                # Cache the result if successful
                if prediction:
                    self.cache_inference_result(digit_array, prediction)
            
            with self.prediction_lock:
                # Remove from pending
                if stable_key in self.pending_predictions:
                    del self.pending_predictions[stable_key]
                
                if prediction:
                    self.detection_to_prediction[stable_key] = {
                        'prediction': prediction,
                        'timestamp': time.time(),
                        'bbox': bbox,
                        'center': center,
                        'last_seen': time.time()  # Track when this prediction was last seen
                    }
        except Exception as e:
            print(f"Inference error for {stable_key}: {e}")
            # Clean up on error
            with self.prediction_lock:
                if stable_key in self.pending_predictions:
                    del self.pending_predictions[stable_key]
        finally:
            # Clean up temp file
            if temp_file_path:
                try:
                    import os
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
                except Exception as e:
                    print(f"Failed to clean up temp file {temp_file_path}: {e}")
            
            # Always clean up the future
            # Note: We can't access the future directly here, but cleanup_stuck_futures will handle it
    
    def cleanup_old_predictions(self):
        """Clean up old predictions and pending states"""
        current_time = time.time()
        
        with self.prediction_lock:
            # Clean up very old predictions (only remove if not seen for 30 seconds)
            keys_to_remove = []
            for key, pred_data in self.detection_to_prediction.items():
                last_seen = pred_data.get('last_seen', pred_data['timestamp'])
                if current_time - last_seen > 30.0:  # Remove predictions not seen for 30 seconds
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.detection_to_prediction[key]
            
            # Clean up stuck pending predictions
            pending_to_remove = []
            for key, pending_data in self.pending_predictions.items():
                if current_time - pending_data['timestamp'] > self.inference_timeout + 2.0:  # Give extra time
                    pending_to_remove.append(key)
            
            for key in pending_to_remove:
                del self.pending_predictions[key]
            
            # Clean up old inference times
            old_inference_times = []
            for key, last_time in self.last_inference_time.items():
                if current_time - last_time > 30.0:  # Remove entries older than 30 seconds
                    old_inference_times.append(key)
            
            for key in old_inference_times:
                del self.last_inference_time[key]
    
    def prediction_worker(self):
        """Worker thread for running predictions"""
        while self.detection_camera.running:
            try:
                # Periodic cleanup of old predictions and stuck futures
                current_time = time.time()
                if current_time - self.last_cleanup_time > self.prediction_cleanup_interval:
                    self.cleanup_old_predictions()
                    self.cleanup_stuck_futures()
                    self.last_cleanup_time = current_time
                
                detections = self.detection_camera.get_latest_detections()
                
                for det in detections:
                    if det['confidence'] > 0.5:
                        # Create a stable key based on position and size
                        stable_key = self.create_stable_key(det['bbox'])
                        x, y, w, h = det['bbox']
                        center_x, center_y = x + w//2, y + h//2
                        
                        # Check if we already have a prediction for this area or if it's pending
                        should_process = False
                        
                        with self.prediction_lock:
                            if stable_key not in self.detection_to_prediction:
                                if stable_key not in self.pending_predictions:
                                    # Check cooldown to prevent rapid re-inference
                                    last_time = self.last_inference_time.get(stable_key, 0)
                                    if current_time - last_time > self.inference_cooldown:
                                        should_process = True
                                else:
                                    # Check if pending prediction is stuck
                                    pending_time = current_time - self.pending_predictions[stable_key]['timestamp']
                                    if pending_time > self.inference_timeout + 1.0:  # Give extra time before retry
                                        # Remove stuck pending prediction and retry
                                        del self.pending_predictions[stable_key]
                                        should_process = True
                        
                        if should_process:
                            # Mark as pending
                            with self.prediction_lock:
                                self.pending_predictions[stable_key] = {
                                    'timestamp': time.time(),
                                    'bbox': det['bbox'],
                                    'center': (center_x, center_y)
                                }
                            
                            # Submit inference to thread pool with timeout
                            try:
                                # Check if we have too many active inferences (prevent resource exhaustion)
                                if len(self.active_futures) >= 3:  # Max 3 concurrent inferences
                                    # Skip this inference to prevent resource exhaustion
                                    continue
                                
                                # Record inference time for throttling
                                self.last_inference_time[stable_key] = current_time
                                
                                future = self.inference_executor.submit(
                                    self._run_inference_async, 
                                    stable_key, 
                                    det['preprocessed_digit'], 
                                    det['bbox'], 
                                    (center_x, center_y)
                                )
                                # Track the future
                                self.active_futures.add(future)
                                
                                # Don't wait for result here - let it run asynchronously
                            except Exception as e:
                                print(f"Failed to submit inference: {e}")
                                # Clean up pending prediction if submission failed
                                with self.prediction_lock:
                                    if stable_key in self.pending_predictions:
                                        del self.pending_predictions[stable_key]
                
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Prediction worker error: {e}")
                time.sleep(0.1)
    
    def draw_prediction_overlay(self, frame, detections):
        """Draw detection results with predictions on frame"""
        overlay = frame.copy()
        current_time = time.time()
        
        # First, update last_seen for any predictions that match current detections
        with self.prediction_lock:
            for det in detections:
                if det['confidence'] > 0.5:
                    stable_key = self.create_stable_key(det['bbox'])
                    if stable_key in self.detection_to_prediction:
                        # Update last_seen time to keep prediction alive
                        self.detection_to_prediction[stable_key]['last_seen'] = current_time
        
        # Draw all detections and their predictions
        for det in detections:
            if det['confidence'] > 0.5:
                x, y, w, h = det['bbox']
                
                # Create stable key for this detection (must match prediction_worker)
                stable_key = self.create_stable_key(det['bbox'])
                
                # Determine box color based on detection confidence
                if det['confidence'] > 0.8:
                    box_color = (0, 255, 0)  # Green - high confidence
                elif det['confidence'] > 0.6:
                    box_color = (0, 255, 255)  # Yellow - medium confidence
                else:
                    box_color = (0, 165, 255)  # Orange - low confidence
                
                # Draw bounding box
                cv2.rectangle(overlay, (x, y), (x + w, y + h), box_color, 2)
                
                # Check prediction status using stable key
                with self.prediction_lock:
                    if stable_key in self.detection_to_prediction:
                        pred_data = self.detection_to_prediction[stable_key]
                        
                        # Show prediction result - keep it visible as long as last_seen is recent
                        digit = pred_data['prediction']['digit']
                        label = str(digit)
                        
                        # Draw prediction text - just black text
                        cv2.putText(overlay, label, (x, y + h + 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)  # Black text
                    else:
                        # Try to find a nearby prediction if no direct match (if enabled)
                        if self.use_fallback_matching:
                            best_match = self.find_best_matching_prediction(det['bbox'])
                            if best_match and best_match in self.detection_to_prediction:
                                pred_data = self.detection_to_prediction[best_match]
                                digit = pred_data['prediction']['digit']
                                label = str(digit)
                                
                                # Draw prediction text - just black text
                                cv2.putText(overlay, label, (x, y + h + 30), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)  # Black text
                                
                                # Update last_seen for the matched prediction
                                pred_data['last_seen'] = current_time
        
        return overlay
    
    def start_recognition(self):
        """Start the digit recognition system"""
        print("🎥 Starting Digit Recognition Camera - Press 'q' to quit")
        
        # Start detection camera
        self.detection_camera.running = True
        
        # Start threads
        capture_thread = threading.Thread(target=self.detection_camera.capture_frames, daemon=True)
        detection_thread = threading.Thread(target=self.detection_camera.detection_worker, daemon=True)
        prediction_thread = threading.Thread(target=self.prediction_worker, daemon=True)
        
        capture_thread.start()
        detection_thread.start()
        prediction_thread.start()
        
        # Create window
        cv2.namedWindow("Digit Recognition", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Digit Recognition", 1280, 720)
        
        try:
            while True:
                try:
                    frame = self.detection_camera.frame_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                detections = self.detection_camera.get_latest_detections()
                display_frame = self.draw_prediction_overlay(frame, detections)
                
                cv2.imshow("Digit Recognition", display_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
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

def main():
    camera = DigitRecognitionCamera()
    camera.start_recognition()

if __name__ == "__main__":
    main()
