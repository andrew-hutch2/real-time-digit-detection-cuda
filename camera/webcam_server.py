#!/usr/bin/env python3
"""
High-Quality Webcam MJPEG Server for Windows
Optimized for better image quality and performance.
"""

from flask import Flask, Response
import cv2
import time

app = Flask(__name__)

# Initialize camera with high-quality settings
camera = cv2.VideoCapture(0)

# Set camera properties for better quality
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)   # Higher resolution
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Higher resolution
camera.set(cv2.CAP_PROP_FPS, 30)             # Target FPS
camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)       # Reduce buffer lag
camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)        # Enable autofocus
camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)    # Enable auto exposure

# Get actual camera properties
width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = camera.get(cv2.CAP_PROP_FPS)

print(f"📐 Camera resolution: {width}x{height}")
print(f"🎬 Camera FPS: {fps}")

@app.route('/video')
def video():
    def generate():
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = camera.read()
            if not ret:
                print("⚠️ Failed to read frame from camera")
                break
            
            # Optional: Add timestamp overlay
            timestamp = time.strftime("%H:%M:%S")
            cv2.putText(frame, timestamp, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Encode with high quality JPEG settings
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]  # High quality (95/100)
            _, buffer = cv2.imencode('.jpg', frame, encode_param)
            frame_bytes = buffer.tobytes()
            
            # Yield frame in MJPEG format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            frame_count += 1
            
            # Small delay to prevent overwhelming the stream
            time.sleep(0.01)
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    return {
        'status': 'running',
        'resolution': f"{width}x{height}",
        'fps': fps,
        'quality': 'high (95/100)'
    }

@app.route('/')
def index():
    return f"""
    <html>
    <head><title>High-Quality Webcam Stream</title></head>
    <body>
        <h1>High-Quality Webcam Stream</h1>
        <p>Resolution: {width}x{height}</p>
        <p>FPS: {fps}</p>
        <p>Quality: High (95/100)</p>
        <img src="/video" style="max-width: 100%; height: auto;">
    </body>
    </html>
    """

if __name__ == '__main__':
    print("🚀 Starting high-quality MJPEG server...")
    print("📺 Stream URL: http://0.0.0.0:8080/video")
    print("🌐 Web interface: http://0.0.0.0:8080/")
    print("Press Ctrl+C to stop")
    
    try:
        app.run(host='0.0.0.0', port=8080, threaded=True, debug=False)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")
    finally:
        camera.release()
        print("✅ Camera released")
