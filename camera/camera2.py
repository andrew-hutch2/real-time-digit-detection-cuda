#!/usr/bin/env python3
"""
Advanced Camera-Based Digit Recognition System
Features:
- Up to 10 concurrent digit inferences
- Hungarian algorithm for optimal digit assignment
- Improved tracking with Kalman filtering
- Better error handling and performance optimization
"""

import cv2
import numpy as np
import time
import threading
import queue
import hashlib
import subprocess
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set
import json
from datetime import datetime

# Try to import scipy for Hungarian algorithm, fallback to simple assignment
try:
    from scipy.optimize import linear_sum_assignment
    HUNGARIAN_AVAILABLE = True
except ImportError:
    HUNGARIAN_AVAILABLE = False
    print("Warning: scipy not available, using simple assignment algorithm")

@dataclass
class DigitDetection:
    """Represents a detected digit with tracking information"""
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    center: Tuple[int, int]
    digit_array: np.ndarray
    confidence: float
    timestamp: float
    stable_key: str
    track_id: Optional[int] = None
    prediction: Optional[int] = None
    prediction_confidence: Optional[float] = None
    inference_id: Optional[str] = None

@dataclass
class TrackedDigit:
    """Represents a tracked digit over time"""
    track_id: int
    detections: deque
    last_seen: float
    prediction_history: deque
    confidence_history: deque
    stable_count: int = 0
    is_stable: bool = False

class KalmanTracker:
    """Simple Kalman filter for digit tracking"""
    
    def __init__(self, initial_center, track_id):
        self.track_id = track_id
        self.kf = cv2.KalmanFilter(4, 2)  # 4 state variables, 2 measurements
        
        # State transition matrix (constant velocity model)
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Measurement matrix
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # Process noise covariance
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        
        # Measurement noise covariance
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
        
        # Error covariance
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        
        # Initialize state
        self.kf.statePre = np.array([initial_center[0], initial_center[1], 0, 0], dtype=np.float32)
        self.kf.statePost = np.array([initial_center[0], initial_center[1], 0, 0], dtype=np.float32)
        
        self.last_update = time.time()
        self.predicted_center = initial_center
    
    def predict(self):
        """Predict next position"""
        prediction = self.kf.predict()
        self.predicted_center = (int(prediction[0]), int(prediction[1]))
        return self.predicted_center
    
    def update(self, measurement):
        """Update with new measurement"""
        self.kf.correct(np.array(measurement, dtype=np.float32))
        self.last_update = time.time()
        return (int(self.kf.statePost[0]), int(self.kf.statePost[1]))

class AdvancedDigitTracker:
    """Advanced digit tracker with Hungarian algorithm assignment"""
    
    def __init__(self, max_tracks=10, max_disappeared=30, min_stable_detections=5):
        self.max_tracks = max_tracks
        self.max_disappeared = max_disappeared
        self.min_stable_detections = min_stable_detections
        self.next_track_id = 0
        
        # Active tracks
        self.tracks: Dict[int, TrackedDigit] = {}
        self.kalman_trackers: Dict[int, KalmanTracker] = {}
        
        # Assignment cost parameters
        self.position_weight = 1.0
        self.size_weight = 0.5
        self.confidence_weight = 0.3
        
        # Tracking parameters
        self.max_distance = 100  # Maximum distance for assignment
        self.stable_threshold = 0.8  # Confidence threshold for stable predictions
    
    def _calculate_cost_matrix(self, detections: List[DigitDetection], tracks: Dict[int, TrackedDigit]) -> np.ndarray:
        """Calculate cost matrix for Hungarian algorithm assignment"""
        if not detections or not tracks:
            return np.array([])
        
        n_detections = len(detections)
        n_tracks = len(tracks)
        cost_matrix = np.full((n_detections, n_tracks), float('inf'))
        
        track_ids = list(tracks.keys())
        
        for i, detection in enumerate(detections):
            for j, track_id in enumerate(track_ids):
                track = tracks[track_id]
                if not track.detections:
                    continue
                
                # Get last detection from track
                last_detection = track.detections[-1]
                
                # Calculate position distance
                pos_distance = np.sqrt(
                    (detection.center[0] - last_detection.center[0])**2 +
                    (detection.center[1] - last_detection.center[1])**2
                )
                
                # Calculate size similarity
                size_similarity = 1.0 - abs(
                    (detection.bbox[2] * detection.bbox[3]) - 
                    (last_detection.bbox[2] * last_detection.bbox[3])
                ) / max(detection.bbox[2] * detection.bbox[3], last_detection.bbox[2] * last_detection.bbox[3])
                
                # Calculate confidence similarity
                conf_similarity = 1.0 - abs(detection.confidence - last_detection.confidence)
                
                # Combined cost (lower is better)
                if pos_distance <= self.max_distance:
                    cost = (
                        self.position_weight * pos_distance +
                        self.size_weight * (1.0 - size_similarity) * 100 +
                        self.confidence_weight * (1.0 - conf_similarity) * 100
                    )
                    cost_matrix[i, j] = cost
        
        return cost_matrix
    
    def _assign_detections_to_tracks(self, detections: List[DigitDetection]) -> Tuple[Dict[int, int], List[int], List[int]]:
        """Assign detections to tracks using Hungarian algorithm or simple assignment"""
        if not detections:
            return {}, [], []
        
        # Filter active tracks
        active_tracks = {tid: track for tid, track in self.tracks.items() 
                        if time.time() - track.last_seen < self.max_disappeared}
        
        if not active_tracks:
            return {}, list(range(len(detections))), []
        
        if HUNGARIAN_AVAILABLE and len(detections) > 1 and len(active_tracks) > 1:
            # Use Hungarian algorithm
            cost_matrix = self._calculate_cost_matrix(detections, active_tracks)
            if cost_matrix.size > 0:
                row_indices, col_indices = linear_sum_assignment(cost_matrix)
                
                # Filter out assignments with infinite cost
                valid_assignments = {}
                assigned_detections = set()
                assigned_tracks = set()
                
                for row, col in zip(row_indices, col_indices):
                    if cost_matrix[row, col] != float('inf'):
                        track_id = list(active_tracks.keys())[col]
                        valid_assignments[track_id] = row
                        assigned_detections.add(row)
                        assigned_tracks.add(track_id)
                
                unassigned_detections = [i for i in range(len(detections)) if i not in assigned_detections]
                unassigned_tracks = [tid for tid in active_tracks.keys() if tid not in assigned_tracks]
                
                return valid_assignments, unassigned_detections, unassigned_tracks
        
        # Fallback to simple nearest neighbor assignment
        return self._simple_assignment(detections, active_tracks)
    
    def _simple_assignment(self, detections: List[DigitDetection], tracks: Dict[int, TrackedDigit]) -> Tuple[Dict[int, int], List[int], List[int]]:
        """Simple nearest neighbor assignment as fallback"""
        assignments = {}
        assigned_detections = set()
        assigned_tracks = set()
        
        for track_id, track in tracks.items():
            if not track.detections:
                continue
            
            last_detection = track.detections[-1]
            best_distance = float('inf')
            best_detection_idx = -1
            
            for i, detection in enumerate(detections):
                if i in assigned_detections:
                    continue
                
                distance = np.sqrt(
                    (detection.center[0] - last_detection.center[0])**2 +
                    (detection.center[1] - last_detection.center[1])**2
                )
                
                if distance < best_distance and distance <= self.max_distance:
                    best_distance = distance
                    best_detection_idx = i
            
            if best_detection_idx != -1:
                assignments[track_id] = best_detection_idx
                assigned_detections.add(best_detection_idx)
                assigned_tracks.add(track_id)
        
        unassigned_detections = [i for i in range(len(detections)) if i not in assigned_detections]
        unassigned_tracks = [tid for tid in tracks.keys() if tid not in assigned_tracks]
        
        return assignments, unassigned_detections, unassigned_tracks
    
    def update(self, detections: List[DigitDetection]) -> List[DigitDetection]:
        """Update tracks with new detections"""
        current_time = time.time()
        
        # Assign detections to existing tracks
        assignments, unassigned_detections, unassigned_tracks = self._assign_detections_to_tracks(detections)
        
        # Update assigned tracks
        for track_id, detection_idx in assignments.items():
            detection = detections[detection_idx]
            
            # Update Kalman filter
            if track_id in self.kalman_trackers:
                self.kalman_trackers[track_id].update(detection.center)
            else:
                self.kalman_trackers[track_id] = KalmanTracker(detection.center, track_id)
            
            # Update track
            track = self.tracks[track_id]
            track.detections.append(detection)
            track.last_seen = current_time
            
            # Keep only recent detections
            while len(track.detections) > 20:
                track.detections.popleft()
            
            # Update prediction history if available
            if detection.prediction is not None:
                track.prediction_history.append(detection.prediction)
                track.confidence_history.append(detection.prediction_confidence or 0.0)
                
                # Keep only recent predictions
                while len(track.prediction_history) > 10:
                    track.prediction_history.popleft()
                    track.confidence_history.popleft()
                
                # Check if track is stable
                if len(track.prediction_history) >= self.min_stable_detections:
                    recent_predictions = list(track.prediction_history)[-self.min_stable_detections:]
                    recent_confidences = list(track.confidence_history)[-self.min_stable_detections:]
                    
                    # Check if predictions are consistent and confident
                    if (len(set(recent_predictions)) == 1 and 
                        all(conf > self.stable_threshold for conf in recent_confidences)):
                        track.is_stable = True
                        track.stable_count += 1
                    else:
                        track.is_stable = False
                        track.stable_count = 0
            
            # Assign track ID to detection
            detection.track_id = track_id
        
        # Create new tracks for unassigned detections
        for detection_idx in unassigned_detections:
            if len(self.tracks) < self.max_tracks:
                detection = detections[detection_idx]
                track_id = self.next_track_id
                self.next_track_id += 1
                
                # Create new track
                track = TrackedDigit(
                    track_id=track_id,
                    detections=deque([detection]),
                    last_seen=current_time,
                    prediction_history=deque(),
                    confidence_history=deque()
                )
                
                # Create Kalman tracker
                self.kalman_trackers[track_id] = KalmanTracker(detection.center, track_id)
                
                # Add to tracks
                self.tracks[track_id] = track
                detection.track_id = track_id
        
        # Remove old tracks
        tracks_to_remove = []
        for track_id, track in self.tracks.items():
            if current_time - track.last_seen > self.max_disappeared:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
            if track_id in self.kalman_trackers:
                del self.kalman_trackers[track_id]
        
        # Return detections with track IDs
        return detections

class AdvancedCameraProcessor:
    """Advanced camera processor with concurrent inference and improved tracking"""
    
    def __init__(self, camera_url="http://192.168.1.100:8080/video", max_concurrent_inferences=10):
        self.camera_url = camera_url
        self.max_concurrent_inferences = max_concurrent_inferences
        
        # Initialize detection system
        from detectDigit import MSERDigitDetector
        self.detector = MSERDigitDetector()
        
        # Initialize tracker
        self.tracker = AdvancedDigitTracker(max_tracks=15)
        
        # Inference system
        self.inference_binary = "../bin/inference"
        self.inference_timeout = 2.0
        
        # Thread pool for concurrent inference
        self.inference_executor = ThreadPoolExecutor(
            max_workers=max_concurrent_inferences, 
            thread_name_prefix="inference"
        )
        
        # Active inference futures
        self.active_futures: Dict[str, any] = {}
        
        # Inference cache
        self.inference_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Performance tracking
        self.inference_times = deque(maxlen=100)
        self.detection_counts = deque(maxlen=100)
        
        # Results display - use camera.py approach
        self.detection_to_prediction = {}  # Map stable_key to prediction data
        self.prediction_lock = threading.Lock()  # Thread safety for predictions
        
        print(f"Advanced Camera Processor initialized")
        print(f"Max concurrent inferences: {max_concurrent_inferences}")
        print(f"Hungarian algorithm available: {HUNGARIAN_AVAILABLE}")
    
    def create_stable_key(self, bbox):
        """Create a more stable key for tracking digits (from camera.py)"""
        x, y, w, h = bbox
        center_x, center_y = x + w//2, y + h//2
        
        # Use coarser granularity to reduce flickering
        # This makes the key more stable for moving digits
        grid_x = center_x // 20  # 20 pixel grid
        grid_y = center_y // 20  # 20 pixel grid
        size_bucket = (w + h) // 40  # Size bucket (roughly 40 pixel increments)
        
        return f"{grid_x}_{grid_y}_{size_bucket}"
    
    def _generate_inference_id(self, detection: DigitDetection) -> str:
        """Generate unique ID for inference request"""
        return f"{detection.track_id}_{int(detection.timestamp * 1000)}"
    
    def _check_inference_cache(self, digit_array: np.ndarray) -> Optional[int]:
        """Check if we have cached inference result for this digit"""
        # Create hash of digit data
        digit_hash = hashlib.md5(digit_array.tobytes()).hexdigest()
        
        if digit_hash in self.inference_cache:
            self.cache_hits += 1
            return self.inference_cache[digit_hash]
        
        self.cache_misses += 1
        return None
    
    def _cache_inference_result(self, digit_array: np.ndarray, prediction: int):
        """Cache inference result"""
        digit_hash = hashlib.md5(digit_array.tobytes()).hexdigest()
        self.inference_cache[digit_hash] = prediction
        
        # Limit cache size
        if len(self.inference_cache) > 1000:
            # Remove oldest entries
            keys_to_remove = list(self.inference_cache.keys())[:100]
            for key in keys_to_remove:
                del self.inference_cache[key]
    
    def _create_temp_file(self, detection: DigitDetection) -> str:
        """Create temporary file for inference"""
        temp_fd, temp_path = tempfile.mkstemp(suffix='.bin', prefix=f'digit_{detection.track_id}_')
        os.close(temp_fd)
        
        # Save digit data
        if not self.detector.save_digit_binary(detection.digit_array, temp_path):
            os.unlink(temp_path)
            return None
        
        return temp_path
    
    def _run_cuda_inference(self, detection: DigitDetection, temp_file_path: str) -> Optional[int]:
        """Run CUDA inference on a single digit"""
        try:
            # Run CUDA inference
            cmd = [self.inference_binary, temp_file_path, "retrained_model_best250epoch.bin"]
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=self.inference_timeout, 
                cwd="../bin"
            )
            
            if result.returncode == 0:
                # Parse prediction from output
                output_lines = result.stdout.strip().split('\n')
                for line in output_lines:
                    if 'Prediction:' in line:
                        try:
                            prediction = int(line.split('Prediction:')[1].strip().split()[0])
                            return prediction
                        except (ValueError, IndexError):
                            continue
            else:
                print(f"Inference failed for track {detection.track_id}: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print(f"Inference timeout for track {detection.track_id}")
        except Exception as e:
            print(f"Inference error for track {detection.track_id}: {e}")
        
        return None
    
    def _run_inference_async(self, detection: DigitDetection) -> Optional[int]:
        """Run inference asynchronously for a single detection"""
        # Check cache first
        cached_prediction = self._check_inference_cache(detection.digit_array)
        if cached_prediction is not None:
            return cached_prediction
        
        # Create temp file
        temp_file_path = self._create_temp_file(detection)
        if not temp_file_path:
            return None
        
        try:
            # Run inference
            prediction = self._run_cuda_inference(detection, temp_file_path)
            
            # Cache result
            if prediction is not None:
                self._cache_inference_result(detection.digit_array, prediction)
            
            return prediction
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    def _submit_inference(self, detection: DigitDetection) -> str:
        """Submit inference request and return inference ID"""
        inference_id = self._generate_inference_id(detection)
        
        # Submit to thread pool
        future = self.inference_executor.submit(self._run_inference_async, detection)
        self.active_futures[inference_id] = {
            'future': future,
            'detection': detection,
            'timestamp': time.time()
        }
        
        return inference_id
    
    def _process_completed_inferences(self) -> List[Tuple[DigitDetection, int]]:
        """Process completed inference requests"""
        results = []
        completed_futures = []
        
        for inference_id, inference_data in self.active_futures.items():
            future = inference_data['future']
            
            if future.done():
                try:
                    prediction = future.result()
                    detection = inference_data['detection']
                    
                    if prediction is not None:
                        detection.prediction = prediction
                        detection.prediction_confidence = 0.95  # Assume high confidence for now
                        results.append((detection, prediction))
                        
                        # Track inference time
                        inference_time = time.time() - inference_data['timestamp']
                        self.inference_times.append(inference_time)
                    
                except Exception as e:
                    print(f"Inference error for {inference_id}: {e}")
                
                completed_futures.append(inference_id)
        
        # Remove completed futures
        for inference_id in completed_futures:
            del self.active_futures[inference_id]
        
        return results
    
    def _update_displayed_results(self, results: List[Tuple[DigitDetection, int]]):
        """Update the displayed results with new inference results (camera.py approach)"""
        current_time = time.time()
        
        with self.prediction_lock:
            for detection, prediction in results:
                if detection.track_id is not None:
                    # Create stable key for this detection
                    stable_key = self.create_stable_key(detection.bbox)
                    
                    # Store prediction in persistent dictionary
                    self.detection_to_prediction[stable_key] = {
                        'prediction': {'digit': prediction},
                        'timestamp': current_time,
                        'bbox': detection.bbox,
                        'center': detection.center,
                        'last_seen': current_time
                    }
                    
                    # Print to console
                    print(f"🎯 INFERENCE RESULT: Track {detection.track_id} → DIGIT {prediction} (Confidence: {detection.prediction_confidence or 0.0:.2f})")
            
            # Clean up very old predictions (only remove if not seen for 30 seconds)
            keys_to_remove = []
            for key, pred_data in self.detection_to_prediction.items():
                last_seen = pred_data.get('last_seen', pred_data['timestamp'])
                if current_time - last_seen > 30.0:  # Remove predictions not seen for 30 seconds
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.detection_to_prediction[key]
    
    
    def _cleanup_old_futures(self):
        """Clean up old stuck inference futures"""
        current_time = time.time()
        stuck_futures = []
        
        for inference_id, inference_data in self.active_futures.items():
            if current_time - inference_data['timestamp'] > self.inference_timeout + 2.0:
                stuck_futures.append(inference_id)
        
        for inference_id in stuck_futures:
            future = self.active_futures[inference_id]['future']
            future.cancel()
            del self.active_futures[inference_id]
        
        if stuck_futures:
            print(f"Cleaned up {len(stuck_futures)} stuck inference futures")
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Tuple[DigitDetection, int]]]:
        """Process a single frame and return annotated frame with results"""
        start_time = time.time()
        
        # Detect digits
        detections = self.detector.detect_digits(frame)
        
        # Convert to DigitDetection objects
        digit_detections = []
        for detection_data in detections:
            bbox = detection_data['bbox']
            confidence = detection_data['confidence']
            digit_array = detection_data['preprocessed_digit']
            
            # Calculate center from bbox
            x, y, w, h = bbox
            center = (x + w // 2, y + h // 2)
            
            detection = DigitDetection(
                bbox=bbox,
                center=center,
                digit_array=digit_array,
                confidence=confidence,
                timestamp=time.time(),
                stable_key=f"{center[0]}_{center[1]}"
            )
            
            digit_detections.append(detection)
        
        # Update tracker
        tracked_detections = self.tracker.update(digit_detections)
        
        # Submit new inferences for stable detections without predictions
        for detection in tracked_detections:
            if (detection.track_id is not None and 
                detection.prediction is None and 
                len(self.active_futures) < self.max_concurrent_inferences):
                
                # Check if we already have a pending inference for this track
                has_pending = any(
                    inf_data['detection'].track_id == detection.track_id 
                    for inf_data in self.active_futures.values()
                )
                
                if not has_pending:
                    self._submit_inference(detection)
        
        # Process completed inferences
        inference_results = self._process_completed_inferences()
        
        # Update displayed results
        self._update_displayed_results(inference_results)
        
        # Clean up old futures
        self._cleanup_old_futures()
        
        # Create annotated frame
        annotated_frame = frame.copy()
        
        # Draw detections and results (exactly like camera.py)
        current_time = time.time()
        
        # Update last_seen for existing predictions
        for detection in tracked_detections:
            if detection.confidence > 0.5:
                stable_key = self.create_stable_key(detection.bbox)
                with self.prediction_lock:
                    if stable_key in self.detection_to_prediction:
                        # Update last_seen time to keep prediction alive
                        self.detection_to_prediction[stable_key]['last_seen'] = current_time
        
        # Draw all detections and their predictions
        for detection in tracked_detections:
            if detection.confidence > 0.5:
                x, y, w, h = detection.bbox
                
                # Create stable key for this detection
                stable_key = self.create_stable_key(detection.bbox)
                
                # Determine box color based on detection confidence
                if detection.confidence > 0.8:
                    box_color = (0, 255, 0)  # Green - high confidence
                elif detection.confidence > 0.6:
                    box_color = (0, 255, 255)  # Yellow - medium confidence
                else:
                    box_color = (0, 165, 255)  # Orange - low confidence
                
                # Draw bounding box
                cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), box_color, 2)
                
                # Check prediction status using stable key
                with self.prediction_lock:
                    if stable_key in self.detection_to_prediction:
                        pred_data = self.detection_to_prediction[stable_key]
                        
                        # Show prediction result - keep it visible as long as last_seen is recent
                        digit = pred_data['prediction']['digit']
                        label = str(digit)
                        
                        # Draw prediction text - just black text
                        cv2.putText(annotated_frame, label, (x, y + h + 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)  # Black text
        
        # Draw performance info
        if self.inference_times:
            avg_inference_time = np.mean(self.inference_times)
            fps = 1.0 / (time.time() - start_time) if time.time() - start_time > 0 else 0
            
            info_text = [
                f"FPS: {fps:.1f}",
                f"Active Inferences: {len(self.active_futures)}",
                f"Avg Inference Time: {avg_inference_time:.3f}s",
                f"Cache Hit Rate: {self.cache_hits/(self.cache_hits+self.cache_misses)*100:.1f}%",
                f"Active Tracks: {len(self.tracker.tracks)}"
            ]
            
            for i, text in enumerate(info_text):
                cv2.putText(annotated_frame, text, (10, 30 + i * 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Track detection count
        self.detection_counts.append(len(digit_detections))
        
        return annotated_frame, inference_results
    
    def run(self):
        """Main processing loop"""
        print("Starting advanced camera processing...")
        print("Press 'q' to quit, 's' to save frame, 'r' to reset tracker")
        
        # Open camera
        cap = cv2.VideoCapture(self.camera_url)
        if not cap.isOpened():
            print(f"Error: Could not open camera at {self.camera_url}")
            return
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read frame")
                    break
                
                # Process frame
                annotated_frame, results = self.process_frame(frame)
                
                # Display results
                cv2.imshow('Advanced Digit Recognition', annotated_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"advanced_frame_{timestamp}.jpg"
                    cv2.imwrite(filename, annotated_frame)
                    print(f"Frame saved as {filename}")
                elif key == ord('r'):
                    self.tracker = AdvancedDigitTracker(max_tracks=15)
                    self.inference_cache.clear()
                    with self.prediction_lock:
                        self.detection_to_prediction.clear()
                    print("Tracker, cache, and predictions reset")
                
                # Print results
                if results:
                    for detection, prediction in results:
                        print(f"Track {detection.track_id}: Predicted digit {prediction}")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.inference_executor.shutdown(wait=True)
            print("Advanced camera processing stopped")

def main():
    """Main function"""
    processor = AdvancedCameraProcessor()
    processor.run()

if __name__ == "__main__":
    main()
