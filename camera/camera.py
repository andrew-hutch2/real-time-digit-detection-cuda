#!/usr/bin/env python3
"""
Simple Camera Connection
Connects to MJPEG stream from Windows webcam server.
"""

import cv2
import numpy as np
import time

def main():
    # MJPEG stream URL - change the IP to your Windows host IP
    # You can find this by running: cat /etc/resolv.conf in WSL2
    stream_url = "http://192.168.1.200:8080/video"
    
    print("🎥 Connecting to camera stream...")
    print(f"📡 Stream URL: {stream_url}")
    
    # Connect to the stream with optimized settings
    cap = cv2.VideoCapture(stream_url)
    
    if not cap.isOpened():
        print("❌ Failed to connect to camera stream")
        print("💡 Make sure the Windows webcam server is running on port 8080")
        return
    
    # Optimize stream settings for better quality
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to prevent lag
    cap.set(cv2.CAP_PROP_FPS, 30)        # Set target FPS
    
    # Get stream properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print("✅ Connected to camera stream!")
    print(f"📐 Stream resolution: {width}x{height}")
    print(f"🎬 Stream FPS: {fps}")
    
    # Create window with high-quality settings
    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Camera", 1280, 720)
    
    # Set window properties for better display
    cv2.setWindowProperty("Camera", cv2.WND_PROP_TOPMOST, 0)
    
    print("🎬 Camera started. Press 'q' to quit.")
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        # Clear any buffered frames to prevent lag
        for _ in range(2):
            cap.grab()
        
        ret, frame = cap.read()
        
        if not ret:
            print("⚠️ Failed to read frame")
            break
        
        # Create a clean copy to avoid reference issues
        display_frame = frame.copy()
        
        # Add frame counter and FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        
        # Add info overlay with better visibility
        cv2.putText(display_frame, f"Frame: {frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Resolution: {width}x{height}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display frame with optimized settings
        cv2.imshow("Camera", display_frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("👋 Camera disconnected")

if __name__ == "__main__":
    main()
