#!/usr/bin/env python3
"""
Accurate MSER-based Digit Detection

This module implements an MSER-based digit detector optimized for accuracy
while maintaining reasonable speed. Designed to detect multiple digits of
various sizes and thicknesses.

Key features:
- Multi-scale detection for various digit sizes
- Enhanced MSER parameters for better accuracy
- Improved border weighting
- Better duplicate removal
- Support for thin and thick digits
- Multiple detection passes for comprehensive coverage
"""

import cv2
import numpy as np
import os
import sys
from typing import List, Tuple, Optional
from datetime import datetime

class AccurateMSERDigitDetector:
    def __init__(self):
        # MSER parameters optimized for accuracy and multiple digit detection
        self.mser = cv2.MSER_create(
            delta=4,           # Balanced delta for good detection
            min_area=100,      # Small enough for thin digits
            max_area=50000,    # Large enough for big digits
            max_variation=0.25, # Lower variation for better accuracy
            min_diversity=0.2,  # Balanced diversity
            max_evolution=200,  # More evolution for better detection
            area_threshold=1.01,
            min_margin=0.003,
            edge_blur_size=5   # Standard blur
        )
        
        # Multi-scale detection parameters
        self.detection_scales = [0.5, 0.75, 1.0]  # Multiple scales for different sizes
        
        # Digit filtering parameters - more lenient for various sizes
        self.min_aspect_ratio = 0.15  # Very lenient for thin digits
        self.max_aspect_ratio = 3.0   # Very lenient for wide digits
        self.min_size = 15            # Small minimum for thin digits
        self.max_size = 300           # Large maximum for big digits
        
        # Border weighting parameters
        self.border_margin = 0.1  # 10% margin from edges
        self.center_weight = 1.0
        self.border_weight = 0.2  # Lower weight for border regions
        
        # Duplicate removal parameters
        self.overlap_threshold = 0.3  # 30% overlap threshold
        
    def create_border_weight_mask(self, height: int, width: int) -> np.ndarray:
        """
        Create a sophisticated border weight mask
        """
        mask = np.ones((height, width), dtype=np.float32)
        
        # Calculate border margins
        margin_h = int(height * self.border_margin)
        margin_w = int(width * self.border_margin)
        
        # Create smooth weighting from center to border
        center_y, center_x = height // 2, width // 2
        max_dist = np.sqrt(center_x**2 + center_y**2)
        
        for y in range(height):
            for x in range(width):
                # Distance from center
                dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                
                # Distance from nearest border
                dist_from_border = min(x, y, width - x - 1, height - y - 1)
                
                # Combined weighting
                center_factor = 1.0 - (dist_from_center / max_dist) * 0.2
                border_factor = min(1.0, dist_from_border / min(margin_w, margin_h))
                
                mask[y, x] = center_factor * border_factor
        
        return mask
    
    def preprocess_frame_accurate(self, frame: np.ndarray, scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Accurate preprocessing with optional scaling
        """
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        
        # Apply scaling if needed
        if scale != 1.0:
            new_height = int(gray.shape[0] * scale)
            new_width = int(gray.shape[1] * scale)
            gray_scaled = cv2.resize(gray, (new_width, new_height))
        else:
            gray_scaled = gray
        
        # Apply morphological operations to enhance digit visibility
        # This helps with both thin and thick digits
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        morphed = cv2.morphologyEx(gray_scaled, cv2.MORPH_CLOSE, kernel)
        
        # Gaussian blur for noise reduction
        blurred = cv2.GaussianBlur(morphed, (3, 3), 0)
        
        # Create border weight mask
        border_mask = self.create_border_weight_mask(blurred.shape[0], blurred.shape[1])
        
        return gray, blurred, border_mask
    
    def detect_mser_regions_accurate(self, blurred: np.ndarray, border_mask: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int, int, int], float]]:
        """
        Accurate MSER region detection with comprehensive filtering
        """
        # Detect MSER regions
        regions, _ = self.mser.detectRegions(blurred)
        
        digit_regions = []
        
        for region in regions:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(region.reshape(-1, 1, 2))
            
            # Calculate properties
            area = cv2.contourArea(region.reshape(-1, 1, 2))
            aspect_ratio = w / h if h > 0 else 0
            
            # Comprehensive filtering
            if (self.min_size < w < self.max_size and 
                self.min_size < h < self.max_size and
                self.min_aspect_ratio < aspect_ratio < self.max_aspect_ratio):
                
                # Calculate weight score
                if y + h <= border_mask.shape[0] and x + w <= border_mask.shape[1]:
                    weight_score = np.mean(border_mask[y:y+h, x:x+w])
                else:
                    weight_score = 0
                
                # Lenient threshold for better detection
                if weight_score > 0.2:
                    # Create mask
                    mask = np.zeros(blurred.shape, dtype=np.uint8)
                    cv2.fillPoly(mask, [region.reshape(-1, 1, 2)], 255)
                    
                    digit_regions.append((mask, (x, y, w, h), weight_score))
        
        # Sort by weight score
        digit_regions.sort(key=lambda x: x[2], reverse=True)
        
        return digit_regions
    
    def remove_duplicates_accurate(self, digit_regions: List[Tuple[np.ndarray, Tuple[int, int, int, int], float]]) -> List[Tuple[np.ndarray, Tuple[int, int, int, int], float]]:
        """
        Accurate duplicate removal with overlap calculation
        """
        if not digit_regions:
            return []
        
        filtered_regions = []
        
        for mask, bbox, weight_score in digit_regions:
            x1, y1, w1, h1 = bbox
            
            # Check overlap with already selected regions
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
                
                # Check if overlap exceeds threshold
                if overlap_area > self.overlap_threshold * min(area1, area2):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered_regions.append((mask, bbox, weight_score))
        
        return filtered_regions
    
    def extract_digit_region_accurate(self, frame: np.ndarray, rect: Tuple[int, int, int, int], scale: float = 1.0) -> np.ndarray:
        """
        Accurate digit region extraction with proper scaling and validation
        """
        x, y, w, h = rect
        
        # Scale back to original size if needed
        if scale != 1.0:
            x = int(x / scale)
            y = int(y / scale)
            w = int(w / scale)
            h = int(h / scale)
        
        # Add padding
        padding = 8
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(frame.shape[1] - x, w + 2 * padding)
        h = min(frame.shape[0] - y, h + 2 * padding)
        
        # Validate coordinates
        if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
            # Return a default 28x28 black image if coordinates are invalid
            return np.zeros((28, 28), dtype=np.uint8)
        
        # Extract region
        digit_roi = frame[y:y+h, x:x+w]
        
        # Validate extracted region
        if digit_roi.size == 0:
            # Return a default 28x28 black image if extraction failed
            return np.zeros((28, 28), dtype=np.uint8)
        
        return digit_roi
    
    def preprocess_digit_accurate(self, digit_roi: np.ndarray) -> np.ndarray:
        """
        Accurate digit preprocessing with validation
        """
        # Validate input
        if digit_roi is None or digit_roi.size == 0:
            # Return a default normalized image
            return np.zeros((28, 28), dtype=np.float32)
        
        # Convert to grayscale if needed
        if len(digit_roi.shape) == 3:
            if digit_roi.shape[2] == 3:  # BGR image
                gray = cv2.cvtColor(digit_roi, cv2.COLOR_BGR2GRAY)
            else:
                gray = digit_roi[:,:,0]  # Take first channel
        else:
            gray = digit_roi.copy()
        
        # Validate grayscale image
        if gray is None or gray.size == 0:
            return np.zeros((28, 28), dtype=np.float32)
        
        # Resize to 28x28
        resized = cv2.resize(gray, (28, 28))
        
        # Invert colors
        inverted = cv2.bitwise_not(resized)
        
        # Normalize to [0, 1] range
        normalized = inverted.astype(np.float32) / 255.0
        
        return normalized
    
    def detect_digits_multi_scale(self, frame: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """
        Multi-scale digit detection for various sizes
        """
        # Validate input frame
        if frame is None or frame.size == 0:
            return []
        
        all_detections = []
        
        # Detect at multiple scales
        for scale in self.detection_scales:
            try:
                # Preprocess at this scale
                gray, blurred, border_mask = self.preprocess_frame_accurate(frame, scale)
                
                # Validate preprocessed images
                if blurred is None or blurred.size == 0:
                    continue
                
                # Detect regions at this scale
                mser_regions = self.detect_mser_regions_accurate(blurred, border_mask)
                
                # Remove duplicates at this scale
                filtered_regions = self.remove_duplicates_accurate(mser_regions)
                
                # Process results
                for mask, rect, weight_score in filtered_regions:
                    # Scale rectangle back to original size
                    x, y, w, h = rect
                    if scale != 1.0:
                        x = int(x / scale)
                        y = int(y / scale)
                        w = int(w / scale)
                        h = int(h / scale)
                        rect = (x, y, w, h)
                    
                    # Validate scaled rectangle
                    if w <= 0 or h <= 0 or x < 0 or y < 0:
                        continue
                    
                    # Extract and preprocess digit
                    digit_roi = self.extract_digit_region_accurate(frame, rect, scale)
                    preprocessed = self.preprocess_digit_accurate(digit_roi)
                    
                    all_detections.append((preprocessed, rect, weight_score))
                    
            except Exception as e:
                # Skip this scale if there's an error
                print(f"⚠️ Error processing scale {scale}: {e}")
                continue
        
        # Remove duplicates across all scales
        if all_detections:
            # Sort by weight score
            all_detections.sort(key=lambda x: x[2], reverse=True)
            
            # Remove duplicates across scales
            final_detections = []
            for preprocessed, rect, weight_score in all_detections:
                x1, y1, w1, h1 = rect
                
                # Check overlap with already selected regions
                is_duplicate = False
                for _, existing_rect, _ in final_detections:
                    x2, y2, w2, h2 = existing_rect
                    
                    # Calculate overlap
                    overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                    overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                    overlap_area = overlap_x * overlap_y
                    
                    # Calculate areas
                    area1 = w1 * h1
                    area2 = w2 * h2
                    
                    # Check if overlap exceeds threshold
                    if overlap_area > self.overlap_threshold * min(area1, area2):
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    final_detections.append((preprocessed, rect))
            
            return final_detections
        
        return []
    
    def detect_digits(self, frame: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """
        Main digit detection function using multi-scale approach
        """
        return self.detect_digits_multi_scale(frame)
    
    def visualize_detection(self, frame: np.ndarray, results: List[Tuple[np.ndarray, Tuple[int, int, int, int]]]) -> np.ndarray:
        """
        Visualization with size indicators
        """
        vis_frame = frame.copy()
        
        for i, (preprocessed, rect) in enumerate(results):
            x, y, w, h = rect
            
            # Draw bounding box with color based on size
            if w * h < 1000:  # Small digits
                color = (0, 255, 255)  # Yellow
            elif w * h < 5000:  # Medium digits
                color = (0, 255, 0)    # Green
            else:  # Large digits
                color = (255, 0, 0)    # Red
            
            thickness = 2
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), color, thickness)
            
            # Add label with size info
            label = f"Digit {i+1} ({w}x{h})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Draw label background
            cv2.rectangle(vis_frame, (x, y - label_size[1] - 10), 
                         (x + label_size[0], y), color, -1)
            
            # Draw label text
            cv2.putText(vis_frame, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return vis_frame

def test_accurate_detection():
    """
    Test function for accurate MSER detection
    """
    if len(sys.argv) != 2:
        print("Usage: python3 mser_digit_detection_accurate.py <image_file>")
        print("Example: python3 mser_digit_detection_accurate.py test_image.jpg")
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
    
    # Test accurate detection
    detector = AccurateMSERDigitDetector()
    
    # Time the detection
    import time
    start_time = time.time()
    results = detector.detect_digits(frame)
    end_time = time.time()
    
    print(f"Accurate MSER detection completed in {end_time - start_time:.3f} seconds")
    print(f"Found {len(results)} potential digits")
    
    # Visualize results
    vis_frame = detector.visualize_detection(frame, results)
    
    # Show results
    cv2.imshow("Accurate MSER Digit Detection", vis_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("Accurate detection test completed!")

if __name__ == "__main__":
    test_accurate_detection()
