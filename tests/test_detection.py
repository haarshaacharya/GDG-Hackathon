import sys
import os
from pathlib import Path
import numpy as np
from PIL import Image

# Add backend directory to path
backend_path = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(backend_path))

from app.services.deepfake_detector import (
    detect_gemini_and_ai_watermark,
    analyze_eye_openness,
    detect_deepfake_and_ai,
)

def test_real_photo_without_watermark():
    print("1. Testing Real Mobile Photo without watermark (showroom selfie)...")
    img = np.zeros((300, 400, 3), dtype=np.uint8) + 120
    pil_img = Image.fromarray(img)
    
    res = detect_deepfake_and_ai(pil_img, is_live_camera=False)
    print(f"Result for Real Image: {res['result']} (Confidence: {res['confidence']}%, Category: {res['category']})")
    print(f"Signals: {res['signals']}")
    assert res["result"] == "REAL", f"Expected REAL, got {res['result']}"
    print("✅ Real photo test passed!\n")

def test_gemini_ai_watermark_image():
    print("2. Testing AI Image with Gemini / AI Watermark tag...")
    img = Image.new("RGB", (300, 300), color=(150, 120, 100))
    raw_ai_bytes = b"header...Google Gemini Imagen 3 SynthID Watermark...prompt: a realistic photo"
    
    res = detect_deepfake_and_ai(img, raw_bytes=raw_ai_bytes, is_live_camera=False)
    print(f"Result for Gemini Watermark Image: {res['result']} (Confidence: {res['confidence']}%, Category: {res['category']})")
    print(f"Signals: {res['signals']}")
    assert res["result"] == "FAKE", f"Expected FAKE, got {res['result']}"
    print("✅ AI watermark test passed!\n")

def test_live_camera_untouched():
    print("3. Testing Live Camera (Eyes Open = REAL, Eyes Closed = FAKE)...")
    face_open = np.zeros((100, 100, 3), dtype=np.uint8) + 160
    face_open[28:38, 20:36] = [40, 30, 30]
    face_open[28:38, 64:80] = [40, 30, 30]
    
    res_open = detect_deepfake_and_ai(face_open, is_live_camera=True)
    assert res_open["result"] == "REAL", f"Expected REAL for open eyes, got {res_open['result']}"
    
    face_closed = np.zeros((100, 100, 3), dtype=np.uint8) + 160
    face_closed[33:34, 20:36] = [130, 130, 130]
    face_closed[33:34, 64:80] = [130, 130, 130]
    
    res_closed = detect_deepfake_and_ai(face_closed, is_live_camera=True)
    assert res_closed["result"] == "FAKE", f"Expected FAKE for closed eyes, got {res_closed['result']}"
    print("✅ Live camera functionality verified untouched and working!\n")

if __name__ == "__main__":
    print("=" * 60)
    print("Running FakeShield Image Upload & Live Camera Tests")
    print("=" * 60)
    test_real_photo_without_watermark()
    test_gemini_ai_watermark_image()
    test_live_camera_untouched()
    print("=" * 60)
    print("All tests passed successfully! 🚀")
