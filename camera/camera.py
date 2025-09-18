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
from detectDigit import DigitDetectionCamera

class DigitRecognitionCamera:
    def __init__(self, stream_url="http://192.168.1.200:8080/video"):
        self.detection_camera = DigitDetectionCamera(stream_url)
        
        # Backend integration
        self.inference_binary = "../digitsClassification/bin/inference"
        self.temp_digit_file = "../digitsClassification/temp_digit_0.bin"
        self.model_weights = "../digitsClassification/trained_model_weights.bin"
        
        # Prediction results
        self.latest_predictions = {}
        self.pending_predictions = {}  # Track predictions in progress
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        
    def run_cuda_inference(self, digit_array):
        """Run CUDA inference on a preprocessed digit"""
        try:
            print(f"💾 Saving digit to {self.temp_digit_file}")
            # Save digit to binary file
            if not self.detection_camera.detector.save_digit_binary(digit_array, self.temp_digit_file):
                print("❌ Failed to save digit to binary file")
                return None
            
            # Run CUDA inference - only pass the image file (weights are hardcoded in binary)
            cmd = [self.inference_binary, self.temp_digit_file]
            print(f"🚀 Running CUDA inference: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            
            print(f"📊 CUDA result: returncode={result.returncode}, stdout='{result.stdout.strip()}', stderr='{result.stderr.strip()}'")
            
            if result.returncode == 0:
                prediction = result.stdout.strip()
                try:
                    digit_prediction = int(prediction)
                    return {'digit': digit_prediction, 'confidence': 1.0}
                except ValueError:
                    print(f"❌ Failed to parse prediction as integer: '{prediction}'")
                    return None
            else:
                print(f"❌ CUDA inference error: {result.stderr}")
                return None
                
        except (subprocess.TimeoutExpired, Exception) as e:
            print(f"💥 Inference error: {e}")
            return None
    
    def prediction_worker(self):
        """Worker thread for running predictions"""
        while self.detection_camera.running:
            try:
                detections = self.detection_camera.get_latest_detections()
                
                for det in detections:
                    if det['confidence'] > 0.5:
                        bbox_key = f"{det['bbox'][0]}_{det['bbox'][1]}_{det['bbox'][2]}_{det['bbox'][3]}"
                        
                        # If no prediction exists and not already processing
                        if bbox_key not in self.latest_predictions and bbox_key not in self.pending_predictions:
                            print(f"🔍 Starting inference for detection at {det['bbox']}")
                            
                            # Mark as pending
                            self.pending_predictions[bbox_key] = {
                                'timestamp': time.time(),
                                'bbox': det['bbox']
                            }
                            
                            # Run prediction in a separate thread to allow "Detecting..." to show
                            def run_inference_async(bbox_key, digit_array, bbox):
                                prediction = self.run_cuda_inference(digit_array)
                                
                                # Remove from pending
                                if bbox_key in self.pending_predictions:
                                    del self.pending_predictions[bbox_key]
                                
                                if prediction:
                                    print(f"✅ Inference result: {prediction['digit']}")
                                    self.latest_predictions[bbox_key] = {
                                        'prediction': prediction,
                                        'timestamp': time.time(),
                                        'bbox': bbox
                                    }
                                else:
                                    print(f"❌ Inference failed for {bbox_key}")
                            
                            # Start async inference
                            inference_thread = threading.Thread(
                                target=run_inference_async, 
                                args=(bbox_key, det['preprocessed_digit'], det['bbox']),
                                daemon=True
                            )
                            inference_thread.start()
                
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Prediction worker error: {e}")
                time.sleep(0.1)
    
    def draw_prediction_overlay(self, frame, detections):
        """Draw detection results with predictions on frame"""
        overlay = frame.copy()
        current_time = time.time()
        
        for det in detections:
            if det['confidence'] > 0.5:
                x, y, w, h = det['bbox']
                bbox_key = f"{x}_{y}_{w}_{h}"
                
                # Determine box color based on detection confidence
                if det['confidence'] > 0.8:
                    box_color = (0, 255, 0)  # Green - high confidence
                elif det['confidence'] > 0.6:
                    box_color = (0, 255, 255)  # Yellow - medium confidence
                else:
                    box_color = (0, 165, 255)  # Orange - low confidence
                
                # Draw bounding box
                cv2.rectangle(overlay, (x, y), (x + w, y + h), box_color, 2)
                
                # Check prediction status
                if bbox_key in self.latest_predictions:
                    pred_data = self.latest_predictions[bbox_key]
                    
                    if current_time - pred_data['timestamp'] < 2.0:
                        # Show prediction result
                        digit = pred_data['prediction']['digit']
                        label = str(digit)
                        cv2.putText(overlay, label, (x, y + h + 20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                    else:
                        # Prediction too old, remove it
                        del self.latest_predictions[bbox_key]
                        
                elif bbox_key in self.pending_predictions:
                    # Show "Detecting..." while inference is running
                    print(f"🔍 Showing 'Detecting...' for {bbox_key}")
                    cv2.putText(overlay, "Detecting...", (x, y + h + 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
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
            
            # Wait for threads to finish
            capture_thread.join(timeout=2)
            detection_thread.join(timeout=2)
            prediction_thread.join(timeout=2)

def main():
    camera = DigitRecognitionCamera()
    camera.start_recognition()

if __name__ == "__main__":
    main()
