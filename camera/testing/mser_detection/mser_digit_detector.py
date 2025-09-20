#!/usr/bin/env python3
"""
MSER-based Digit Detection and Preprocessing Pipeline

This module implements MSER (Maximally Stable Extremal Regions) for digit detection,
which is specifically designed to find stable regions that remain consistent across
different thresholds, making it ideal for digit recognition without detecting noise edges.

Key features:
- MSER-based region detection instead of contour detection
- Border weighting to devalue screen edges
- Step-by-step visualization of the detection process
- Same interface as the original preprocessing visualizer
"""

import cv2
import numpy as np
import os
import sys
from typing import List, Tuple, Optional
from datetime import datetime

class MSERDigitDetector:
    def __init__(self):
        # MSER parameters optimized for digit detection (balanced for fewer duplicates)
        self.mser = cv2.MSER_create(
            delta=4,           # Delta parameter for stability (increased to reduce duplicates)
            min_area=500,      # Minimum area (reduced for more detections)
            max_area=80000,    # Maximum area (increased for larger digits)
            max_variation=0.3, # Maximum variation for stability (reduced for fewer duplicates)
            min_diversity=0.15, # Minimum diversity between regions (increased for fewer duplicates)
            max_evolution=200, # Maximum evolution steps
            area_threshold=1.01, # Area threshold
            min_margin=0.003,  # Minimum margin
            edge_blur_size=5   # Edge blur size
        )
        
        # Digit filtering parameters
        self.min_aspect_ratio = 0.3
        self.max_aspect_ratio = 1.5
        self.min_size = 30
        self.max_size = 200
        
        # Border weighting parameters
        self.border_margin = 0.1  # 10% margin from edges
        self.center_weight = 1.0
        self.border_weight = 0.3  # Much lower weight for border regions
        
    def create_border_weight_mask(self, height: int, width: int) -> np.ndarray:
        """
        Create a weight mask that devalues regions near the image borders.
        This helps reduce false detections from paper edges and screen borders.
        """
        mask = np.ones((height, width), dtype=np.float32)
        
        # Calculate border margins
        margin_h = int(height * self.border_margin)
        margin_w = int(width * self.border_margin)
        
        # Create distance-based weighting
        center_y, center_x = height // 2, width // 2
        max_dist = np.sqrt(center_x**2 + center_y**2)
        
        for y in range(height):
            for x in range(width):
                # Calculate distance from center
                dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                
                # Calculate distance from nearest border
                dist_from_border = min(
                    x, y, width - x - 1, height - y - 1
                )
                
                # Weight based on distance from center and border
                center_weight = 1.0 - (dist_from_center / max_dist) * 0.3
                border_weight = min(1.0, dist_from_border / min(margin_w, margin_h))
                
                # Combine weights (lower values near borders)
                mask[y, x] = center_weight * border_weight
        
        return mask
    
    def preprocess_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess the input frame for MSER detection
        Returns: (grayscale, blurred, border_weight_mask)
        """
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        
        # Apply Gaussian blur to reduce noise while preserving edges
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Create border weight mask
        height, width = gray.shape
        border_mask = self.create_border_weight_mask(height, width)
        
        return gray, blurred, border_mask
    
    def detect_mser_regions(self, blurred: np.ndarray, border_mask: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int, int, int], float]]:
        """
        Detect MSER regions and filter them for digit-like properties
        Returns: List of (mask, bbox, weight_score)
        """
        # Detect MSER regions
        regions, _ = self.mser.detectRegions(blurred)
        print(f"   MSER detected {len(regions)} raw regions")
        
        # Convert regions to masks and filter
        digit_regions = []
        size_filtered = 0
        aspect_filtered = 0
        weight_filtered = 0
        
        for region in regions:
            # Create mask for this region
            mask = np.zeros(blurred.shape, dtype=np.uint8)
            points = region.reshape(-1, 1, 2)
            cv2.fillPoly(mask, [points], 255)
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(points)
            
            # Calculate properties
            area = cv2.contourArea(points)
            aspect_ratio = w / h if h > 0 else 0
            
            # Filter for digit-like regions
            if not (self.min_size < w < self.max_size and 
                   self.min_size < h < self.max_size):
                size_filtered += 1
                continue
                
            if not (self.min_aspect_ratio < aspect_ratio < self.max_aspect_ratio):
                aspect_filtered += 1
                continue
                
            # Calculate weight score based on border mask
            region_mask = mask[y:y+h, x:x+w]
            border_region = border_mask[y:y+h, x:x+w]
            
            # Weight score is average of border mask values in the region
            weight_score = np.mean(border_region[region_mask > 0]) if np.any(region_mask > 0) else 0
            
            # Only include regions with reasonable weight scores
            if weight_score > 0.2:  # Threshold for border weighting (reduced for more lenient detection)
                digit_regions.append((mask, (x, y, w, h), weight_score))
            else:
                weight_filtered += 1
        
        print(f"   Filtered: {size_filtered} by size, {aspect_filtered} by aspect, {weight_filtered} by weight")
        print(f"   Final: {len(digit_regions)} regions passed all filters")
        
        # Sort by weight score (higher is better - further from borders)
        digit_regions.sort(key=lambda x: x[2], reverse=True)
        
        # Remove overlapping regions (keep the one with highest weight score)
        filtered_regions = []
        for mask, bbox, weight_score in digit_regions:
            x1, y1, w1, h1 = bbox
            
            # Check for overlap with already selected regions
            is_duplicate = False
            for _, existing_bbox, _ in filtered_regions:
                x2, y2, w2, h2 = existing_bbox
                
                # Calculate overlap
                overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                overlap_area = overlap_x * overlap_y
                
                # Calculate areas
                area1 = w1 * h1
                area2 = w2 * h2
                
                # If overlap is more than 50% of either region, consider it a duplicate
                if overlap_area > 0.5 * min(area1, area2):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered_regions.append((mask, bbox, weight_score))
        
        print(f"   After duplicate removal: {len(filtered_regions)} unique regions")
        
        return filtered_regions
    
    def extract_digit_region(self, frame: np.ndarray, rect: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Extract the digit region from the frame with padding
        """
        x, y, w, h = rect
        
        # Add padding
        padding = 8
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
    
    def detect_digits(self, frame: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """
        Main function to detect and preprocess digits using MSER
        Returns: List of (preprocessed_digit, bounding_rect)
        """
        # For large frames, work on a smaller version for detection
        original_height, original_width = frame.shape[:2]
        scale_factor = 0.5 if original_width > 1280 else 1.0
        
        if scale_factor != 1.0:
            small_width = int(original_width * scale_factor)
            small_height = int(original_height * scale_factor)
            small_frame = cv2.resize(frame, (small_width, small_height))
        else:
            small_frame = frame
        
        # Preprocess frame
        gray, blurred, border_mask = self.preprocess_frame(small_frame)
        
        # Detect MSER regions
        mser_regions = self.detect_mser_regions(blurred, border_mask)
        
        results = []
        for i, (mask, rect, weight_score) in enumerate(mser_regions):
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
            cv2.putText(vis_frame, f"MSER {i}", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return vis_frame

def test_mser_detection():
    """
    Test function to demonstrate MSER digit detection
    """
    if len(sys.argv) != 2:
        print("Usage: python3 mser_digit_detector.py <image_file>")
        print("Example: python3 mser_digit_detector.py test_image.jpg")
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
    
    # Detect digits using MSER
    detector = MSERDigitDetector()
    results = detector.detect_digits(frame)
    
    print(f"Found {len(results)} potential digits using MSER")
    
    # Visualize results
    vis_frame = detector.visualize_detection(frame, results)
    
    # Show results
    cv2.imshow("MSER Digit Detection", vis_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("MSER detection test completed!")

if __name__ == "__main__":
    test_mser_detection()
