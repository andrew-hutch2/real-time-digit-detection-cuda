#!/usr/bin/env python3
"""
Fast MSER-based Digit Detection

This module implements a highly optimized MSER-based digit detector focused on speed
while maintaining good detection quality. Uses multiple performance optimizations
including frame downscaling, simplified MSER parameters, and efficient filtering.

Key optimizations:
- Frame downscaling for detection (2x faster)
- Simplified MSER parameters for speed
- Reduced border weighting complexity
- Efficient duplicate removal
- Minimal preprocessing steps
"""

import cv2
import numpy as np
import os
import sys
from typing import List, Tuple, Optional
from datetime import datetime

class FastMSERDigitDetector:
    def __init__(self):
        # Fast MSER parameters optimized for speed
        self.mser = cv2.MSER_create(
            delta=6,           # Higher delta for faster processing
            min_area=200,      # Smaller min area for faster detection
            max_area=20000,    # Smaller max area for faster processing
            max_variation=0.5, # Higher variation for faster processing
            min_diversity=0.3, # Higher diversity for fewer regions
            max_evolution=100, # Reduced evolution steps for speed
            area_threshold=1.01,
            min_margin=0.003,
            edge_blur_size=3   # Smaller blur for speed
        )
        
        # Simplified digit filtering parameters
        self.min_aspect_ratio = 0.2
        self.max_aspect_ratio = 2.0
        self.min_size = 20
        self.max_size = 150
        
        # Simplified border weighting
        self.border_margin = 0.15  # Larger margin for simpler calculation
        self.center_weight = 1.0
        self.border_weight = 0.2
        
        # Performance settings
        self.detection_scale = 0.5  # Process at half resolution for 4x speed improvement
        
    def create_simple_border_mask(self, height: int, width: int) -> np.ndarray:
        """
        Create a simplified border weight mask for speed.
        Uses simple distance calculation instead of complex weighting.
        """
        mask = np.ones((height, width), dtype=np.float32)
        
        # Calculate border margins
        margin_h = int(height * self.border_margin)
        margin_w = int(width * self.border_margin)
        
        # Simple border weighting - just check distance from edges
        for y in range(height):
            for x in range(width):
                # Calculate distance from nearest border
                dist_from_border = min(x, y, width - x - 1, height - y - 1)
                
                # Simple weight: 1.0 in center, 0.2 near borders
                if dist_from_border < min(margin_w, margin_h):
                    mask[y, x] = self.border_weight
                else:
                    mask[y, x] = self.center_weight
        
        return mask
    
    def preprocess_frame_fast(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fast preprocessing optimized for speed
        """
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        
        # Downscale for faster processing
        if self.detection_scale != 1.0:
            small_height = int(gray.shape[0] * self.detection_scale)
            small_width = int(gray.shape[1] * self.detection_scale)
            gray_small = cv2.resize(gray, (small_width, small_height))
        else:
            gray_small = gray
        
        # Minimal blur for speed
        blurred = cv2.GaussianBlur(gray_small, (3, 3), 0)
        
        # Create simple border mask
        border_mask = self.create_simple_border_mask(blurred.shape[0], blurred.shape[1])
        
        return gray, blurred, border_mask
    
    def detect_mser_regions_fast(self, blurred: np.ndarray, border_mask: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int, int, int], float]]:
        """
        Fast MSER region detection with minimal processing
        """
        # Detect MSER regions
        regions, _ = self.mser.detectRegions(blurred)
        
        # Fast filtering with minimal processing
        digit_regions = []
        
        for region in regions:
            # Get bounding rectangle directly
            x, y, w, h = cv2.boundingRect(region.reshape(-1, 1, 2))
            
            # Fast size and aspect ratio filtering
            if (self.min_size < w < self.max_size and 
                self.min_size < h < self.max_size and
                self.min_aspect_ratio < w/h < self.max_aspect_ratio):
                
                # Simple weight calculation
                weight_score = np.mean(border_mask[y:y+h, x:x+w]) if y+h <= border_mask.shape[0] and x+w <= border_mask.shape[1] else 0
                
                # Simple threshold
                if weight_score > 0.3:
                    # Create minimal mask
                    mask = np.zeros(blurred.shape, dtype=np.uint8)
                    cv2.fillPoly(mask, [region.reshape(-1, 1, 2)], 255)
                    
                    digit_regions.append((mask, (x, y, w, h), weight_score))
        
        # Sort by weight score (higher is better)
        digit_regions.sort(key=lambda x: x[2], reverse=True)
        
        return digit_regions
    
    def remove_duplicates_fast(self, digit_regions: List[Tuple[np.ndarray, Tuple[int, int, int, int], float]]) -> List[Tuple[np.ndarray, Tuple[int, int, int, int], float]]:
        """
        Fast duplicate removal using simple overlap checking
        """
        if not digit_regions:
            return []
        
        # Simple approach: keep first (highest weight) and remove overlapping
        filtered_regions = [digit_regions[0]]  # Keep the best one
        
        for mask, bbox, weight_score in digit_regions[1:]:
            x1, y1, w1, h1 = bbox
            
            # Check overlap with already selected regions
            is_duplicate = False
            for _, existing_bbox, _ in filtered_regions:
                x2, y2, w2, h2 = existing_bbox
                
                # Simple overlap check
                if not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered_regions.append((mask, bbox, weight_score))
        
        return filtered_regions
    
    def extract_digit_region_fast(self, frame: np.ndarray, rect: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Fast digit region extraction
        """
        x, y, w, h = rect
        
        # Scale back to original size if needed
        if self.detection_scale != 1.0:
            x = int(x / self.detection_scale)
            y = int(y / self.detection_scale)
            w = int(w / self.detection_scale)
            h = int(h / self.detection_scale)
        
        # Minimal padding
        padding = 4
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(frame.shape[1] - x, w + 2 * padding)
        h = min(frame.shape[0] - y, h + 2 * padding)
        
        # Extract region
        digit_roi = frame[y:y+h, x:x+w]
        
        return digit_roi
    
    def preprocess_digit_fast(self, digit_roi: np.ndarray) -> np.ndarray:
        """
        Fast digit preprocessing
        """
        # Convert to grayscale if needed
        if len(digit_roi.shape) == 3:
            gray = cv2.cvtColor(digit_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = digit_roi.copy()
        
        # Direct resize to 28x28 (no intermediate steps)
        resized = cv2.resize(gray, (28, 28))
        
        # Invert colors
        inverted = cv2.bitwise_not(resized)
        
        # Normalize to [0, 1] range
        normalized = inverted.astype(np.float32) / 255.0
        
        return normalized
    
    def detect_digits(self, frame: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """
        Main fast digit detection function
        """
        # Fast preprocessing
        gray, blurred, border_mask = self.preprocess_frame_fast(frame)
        
        # Fast MSER detection
        mser_regions = self.detect_mser_regions_fast(blurred, border_mask)
        
        # Fast duplicate removal
        filtered_regions = self.remove_duplicates_fast(mser_regions)
        
        # Process results
        results = []
        for mask, rect, weight_score in filtered_regions:
            # Scale rectangle back to original size if needed
            x, y, w, h = rect
            if self.detection_scale != 1.0:
                x = int(x / self.detection_scale)
                y = int(y / self.detection_scale)
                w = int(w / self.detection_scale)
                h = int(h / self.detection_scale)
                rect = (x, y, w, h)
            
            # Extract and preprocess digit
            digit_roi = self.extract_digit_region_fast(frame, rect)
            preprocessed = self.preprocess_digit_fast(digit_roi)
            
            results.append((preprocessed, rect))
        
        return results
    
    def visualize_detection(self, frame: np.ndarray, results: List[Tuple[np.ndarray, Tuple[int, int, int, int]]]) -> np.ndarray:
        """
        Fast visualization
        """
        vis_frame = frame.copy()
        
        for i, (preprocessed, rect) in enumerate(results):
            x, y, w, h = rect
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Add label
            cv2.putText(vis_frame, f"Fast {i}", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return vis_frame

def test_fast_detection():
    """
    Test function for fast MSER detection
    """
    if len(sys.argv) != 2:
        print("Usage: python3 mser_digit_detection_fast.py <image_file>")
        print("Example: python3 mser_digit_detection_fast.py test_image.jpg")
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
    
    # Test fast detection
    detector = FastMSERDigitDetector()
    
    # Time the detection
    import time
    start_time = time.time()
    results = detector.detect_digits(frame)
    end_time = time.time()
    
    print(f"Fast MSER detection completed in {end_time - start_time:.3f} seconds")
    print(f"Found {len(results)} potential digits")
    
    # Visualize results
    vis_frame = detector.visualize_detection(frame, results)
    
    # Show results
    cv2.imshow("Fast MSER Digit Detection", vis_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("Fast detection test completed!")

if __name__ == "__main__":
    test_fast_detection()
