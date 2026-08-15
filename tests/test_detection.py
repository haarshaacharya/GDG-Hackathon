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
    detect_screen_replay_spoof,
    analyze_eye_openness,
    detect_deepfake_and_ai,
)

def test_upload_real_mobile_image():
    print("1. Testing Real Mobile Photo Upload (Selfie / Showroom Click)...")
    # Simulate a compressed mobile JPEG image
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    for y in range(300):
        for x in range(300):
            img[y, x] = [120 + int(25 * np.sin(x / 18.0)), 110, 100 + int(15 * np.cos(y / 18.0))]
    
    pil_real = Image.fromarray(img)
    res = detect_deepfake_and_ai(pil_real, is_live_camera=False)
    print(f"Result for Real Mobile Photo: {res['result']} (Confidence: {res['confidence']}%, Category: {res['category']})")
    print(f"Signals: {res['signals']}")
    assert res["result"] == "REAL", f"Expected REAL, got {res['result']}"
    print("✅ Real mobile photo test passed!\n")

def test_upload_ai_generated_image():
    print("2. Testing AI-Generated Image Upload (Gemini / Midjourney / AI Prompt)...")
    img = Image.new("RGB", (300, 300), color=(140, 110, 95))
    raw_ai_bytes = b"header...parameters: prompt: a cinematic portrait, Model: Midjourney v6, Steps: 30"
    
    res = detect_deepfake_and_ai(img, raw_bytes=raw_ai_bytes, is_live_camera=False)
    print(f"Result for AI Generated Image: {res['result']} (Confidence: {res['confidence']}%, Category: {res['category']})")
    assert res["result"] == "FAKE", f"Expected FAKE, got {res['result']}"
    print("✅ AI generated image test passed!\n")

def test_live_camera_eyes_open_vs_closed():
    print("3. Testing Live Camera Eye Liveness (Open = REAL, Closed = FAKE)...")
    
    # Open eyes
    face_open = np.zeros((100, 100, 3), dtype=np.uint8) + 160
    face_open[28:38, 20:36] = [40, 30, 30]
    face_open[28:38, 64:80] = [40, 30, 30]
    res_open = detect_deepfake_and_ai(face_open, is_live_camera=True)
    assert res_open["result"] == "REAL", f"Expected REAL for open eyes, got {res_open['result']}"
    
    # Closed eyes
    face_closed = np.zeros((100, 100, 3), dtype=np.uint8) + 160
    face_closed[33:34, 20:36] = [130, 130, 130]
    face_closed[33:34, 64:80] = [130, 130, 130]
    res_closed = detect_deepfake_and_ai(face_closed, is_live_camera=True)
    assert res_closed["result"] == "FAKE", f"Expected FAKE for closed eyes, got {res_closed['result']}"
    print("✅ Eye liveness test passed!\n")

def test_live_camera_mobile_screen_replay():
    print("4. Testing Live Camera Mobile Screen Replay Attack...")
    screen_face = np.zeros((120, 120, 3), dtype=np.uint8) + 150
    for y in range(0, 120, 3):
        for x in range(0, 120, 3):
            screen_face[y, x] = np.clip(screen_face[y, x].astype(int) + 65, 0, 255)
    
    screen_face[10:25, 40:65] = [255, 255, 255]
    res_screen = detect_deepfake_and_ai(screen_face, is_live_camera=True)
    assert res_screen["result"] == "FAKE", f"Expected FAKE for screen replay, got {res_screen['result']}"
    print("✅ Mobile screen replay attack test passed!\n")

if __name__ == "__main__":
    print("=" * 65)
    print("Running FakeShield All Test Suites")
    print("=" * 65)
    test_upload_real_mobile_image()
    test_upload_ai_generated_image()
    test_live_camera_eyes_open_vs_closed()
    test_live_camera_mobile_screen_replay()
    print("=" * 65)
    print("All tests passed successfully! 🚀")
