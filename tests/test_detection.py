import sys
import os
from pathlib import Path
import numpy as np
from PIL import Image

# Add backend directory to path
backend_path = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(backend_path))

from app.services.deepfake_detector import (
    detect_gemini_watermark,
    analyze_eye_openness,
    detect_deepfake_and_ai,
)

def test_gemini_watermark_detection():
    print("1. Testing Google Gemini Watermark Detection...")
    img = np.zeros((300, 300, 3), dtype=np.uint8) + 30
    
    # Draw a 4-point Gemini sparkle in bottom-right corner (y: 250..290, x: 250..290)
    cy, cx = 270, 270
    for y in range(250, 290):
        for x in range(250, 290):
            dx = abs(x - cx)
            dy = abs(y - cy)
            # 4-point star ray morphology
            if (dx <= 2 and dy <= 14) or (dy <= 2 and dx <= 14):
                img[y, x] = [240, 130, 80] # Gemini Blue/Purple
            elif dx <= 3 and dy <= 3:
                img[y, x] = [255, 255, 255] # White core

    pil_img = Image.fromarray(img)
    is_gemini, conf, signals = detect_gemini_watermark(img, pil_img)
    print(f"Gemini Watermark Check: is_gemini={is_gemini}, signals={signals}")
    
    res = detect_deepfake_and_ai(img)
    print(f"Prediction result with Gemini watermark: {res['result']} ({res['confidence']}%)")
    assert res["result"] == "FAKE", f"Expected FAKE, got {res['result']}"
    print("✅ Gemini watermark test passed!\n")

def test_live_camera_eyes_open_vs_closed():
    print("2. Testing Live Camera Eye Openness (Open = REAL, Closed = FAKE)...")
    
    # Simulate face crop with OPEN eyes (high vertical contrast, dark pupil)
    face_open = np.zeros((100, 100, 3), dtype=np.uint8) + 160
    # Add dark pupil & sclera in eye band
    face_open[28:38, 20:36] = [40, 30, 30] # Left pupil
    face_open[28:38, 64:80] = [40, 30, 30] # Right pupil
    
    res_open = detect_deepfake_and_ai(face_open, is_live_camera=True)
    print(f"Live Camera with Eyes OPEN: {res_open['result']} ({res_open['confidence']}%), Category: {res_open['category']}")
    assert res_open["result"] == "REAL", f"Expected REAL for open eyes, got {res_open['result']}"
    
    # Simulate face crop with CLOSED eyes (smooth skin, no dark pupil, low vertical contrast)
    face_closed = np.zeros((100, 100, 3), dtype=np.uint8) + 160
    # Only thin eyelid crease
    face_closed[33:34, 20:36] = [130, 130, 130]
    face_closed[33:34, 64:80] = [130, 130, 130]
    
    res_closed = detect_deepfake_and_ai(face_closed, is_live_camera=True)
    print(f"Live Camera with Eyes CLOSED: {res_closed['result']} ({res_closed['confidence']}%), Category: {res_closed['category']}")
    assert res_closed["result"] == "FAKE", f"Expected FAKE for closed eyes, got {res_closed['result']}"
    print("✅ Live camera eyes open/closed test passed!\n")

def test_real_camera_without_gemini():
    print("3. Testing Real Mobile Photo without Gemini Watermark...")
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    for y in range(200):
        for x in range(200):
            img[y, x] = [100 + int(20 * np.sin(x / 15.0)), 120, 110]
    # Sensor noise
    noise = np.random.normal(0, 4.0, (200, 200, 3))
    real_img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    pil_real = Image.fromarray(real_img)
    exif = pil_real.getexif()
    exif[0x010F] = "Samsung"
    exif[0x0110] = "Galaxy S23"
    exif[0x8827] = 100
    
    res = detect_deepfake_and_ai(pil_real, is_live_camera=False)
    print(f"Result for Real Samsung Phone Photo: {res['result']} ({res['confidence']}%)")
    assert res["result"] == "REAL", f"Expected REAL, got {res['result']}"
    print("✅ Real photo test passed!\n")

if __name__ == "__main__":
    print("=" * 60)
    print("Running FakeShield Gemini & Live Eyes Tests")
    print("=" * 60)
    test_gemini_watermark_detection()
    test_live_camera_eyes_open_vs_closed()
    test_real_camera_without_gemini()
    print("=" * 60)
    print("All tests passed successfully! 🚀")
