#!/usr/bin/env python3
"""
Digit Detection and Preprocessing Pipeline for Webcam Frames

This module handles:
1. Detecting digit regions in webcam frames
2. Extracting and preprocessing digits to 28x28 format
3. Converting to the binary format expected by the CUDA model
"""

import cv2
import numpy as np
import os
import sys
from typing import List, Tuple, Optional

class DigitDetector:
    def __init__(self):
        # Optimized parameters for 1080p resolution
        self.min_area = 2000  # Increased for 1080p
        self.max_area = 50000  # Increased for 1080p
        self.min_aspect_ratio = 0.3
        self.max_aspect_ratio = 1.5
        self.min_size = 40  # Increased for 1080p
        
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess the input frame for digit detection
        """
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
            
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive threshold to get binary image
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        return thresh
    
    def find_digit_contours(self, thresh: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """
        Find contours that likely contain digits
        """
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        digit_contours = []
        for contour in contours:
            # Get bounding rectangle
            rect = cv2.boundingRect(contour)
            x, y, w, h = rect
            
            # Calculate area and aspect ratio
            area = cv2.contourArea(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Filter for digit-like regions
            if (self.min_area < area < self.max_area and
                self.min_aspect_ratio < aspect_ratio < self.max_aspect_ratio and
                w > self.min_size and h > self.min_size):
                digit_contours.append((contour, rect))
        
        # Sort by area (largest first)
        digit_contours.sort(key=lambda x: cv2.contourArea(x[0]), reverse=True)
        
        return digit_contours
    
    def extract_digit_region(self, frame: np.ndarray, rect: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Extract the digit region from the frame
        """
        x, y, w, h = rect
        
        # Add some padding
        padding = 5
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(frame.shape[1] - x, w + 2 * padding)
        h = min(frame.shape[0] - y, h + 2 * padding)
        
        # Extract region
        digit_roi = frame[y:y+h, x:x+w]
        
        return digit_roi
    
    def preprocess_digit(self, digit_roi: np.ndarray) -> np.ndarray:
        """
        Preprocess a digit region to 28x28 format for the model
        """
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
    
    def save_digit_binary(self, digit_array: np.ndarray, filename: str) -> bool:
        """
        Save preprocessed digit to binary format for CUDA inference
        """
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
    
    def detect_digits(self, frame: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """
        Main function to detect and preprocess digits in a frame
        
        Returns:
            List of (preprocessed_digit, bounding_rect)
        """
        # For 1080p frames, work on a smaller version for detection, then scale back
        original_height, original_width = frame.shape[:2]
        scale_factor = 0.5  # Process at half resolution for speed
        
        if original_width > 1280:  # Only scale down if frame is large
            small_width = int(original_width * scale_factor)
            small_height = int(original_height * scale_factor)
            small_frame = cv2.resize(frame, (small_width, small_height))
        else:
            small_frame = frame
            scale_factor = 1.0
        
        # Preprocess frame
        thresh = self.preprocess_frame(small_frame)
        
        # Find digit contours
        digit_contours = self.find_digit_contours(thresh)
        
        results = []
        for i, (contour, rect) in enumerate(digit_contours):
            # Scale rectangle back to original size
            x, y, w, h = rect
            if scale_factor != 1.0:
                x = int(x / scale_factor)
                y = int(y / scale_factor)
                w = int(w / scale_factor)
                h = int(h / scale_factor)
                rect = (x, y, w, h)
            
            # Extract digit region from original frame
            digit_roi = self.extract_digit_region(frame, rect)
            
            # Preprocess digit
            preprocessed = self.preprocess_digit(digit_roi)
            
            # Add to results (no temp file needed)
            results.append((preprocessed, rect))
        
        return results
    
    def visualize_detection(self, frame: np.ndarray, results: List[Tuple[np.ndarray, Tuple[int, int, int, int]]]) -> np.ndarray:
        """
        Draw bounding boxes and digit regions on the frame for visualization
        """
        vis_frame = frame.copy()
        
        for i, (preprocessed, rect) in enumerate(results):
            x, y, w, h = rect
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Add label
            cv2.putText(vis_frame, f"Digit {i}", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return vis_frame

def test_digit_detection():
    """
    Test function to demonstrate digit detection
    """
    detector = DigitDetector()
    
    # Create a test image with some digits (you can replace this with actual image loading)
    print("Digit detection test - create a test image with digits to test this function")
    print("Usage: python3 digit_detector.py <image_file>")
    print("Example: python3 digit_detector.py test_image.jpg")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 digit_detector.py <image_file>")
        print("Example: python3 digit_detector.py test_image.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} not found")
        sys.exit(1)
    
    # Load image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not load image {image_path}")
        sys.exit(1)
    
    # Detect digits
    detector = DigitDetector()
    results = detector.detect_digits(frame)
    
    print(f"Found {len(results)} potential digits")
    
    # Visualize results
    vis_frame = detector.visualize_detection(frame, results)
    
    # Show results
    cv2.imshow("Digit Detection", vis_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("Test completed!")
