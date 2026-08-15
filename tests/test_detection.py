import sys
import os
from pathlib import Path
import numpy as np
from PIL import Image

# Add backend directory to path
backend_path = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(backend_path))

from app.services.deepfake_detector import (
    analyze_metadata,
    analyze_frequency_spectrum,
    analyze_chrominance_noise,
    analyze_eye_highlights,
    analyze_ai_studio_background,
    analyze_sensor_noise_and_texture,
    detect_deepfake_and_ai,
)

def test_real_camera_signals():
    print("1. Testing Real Mobile Camera photo simulation...")
    np.random.seed(42)
    base = np.zeros((256, 256, 3), dtype=np.uint8)
    for y in range(256):
        for x in range(256):
            base[y, x] = [120 + int(30 * np.sin(x / 20.0)), 100 + int(20 * np.cos(y / 20.0)), 90]
    
    # Real multi-channel sensor noise in R, G, B
    noise = np.random.normal(0, 5.0, (256, 256, 3))
    real_img = np.clip(base.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    pil_real = Image.fromarray(real_img)
    # Simulate EXIF with Apple iPhone
    exif = pil_real.getexif()
    exif[0x010F] = "Apple"
    exif[0x0110] = "iPhone 14 Pro"
    exif[0x8827] = 100
    exif[0x829D] = (18, 10)
    
    res = detect_deepfake_and_ai(pil_real)
    print(f"Result for Real Mobile Camera: {res['result']} (Confidence: {res['confidence']}%, Category: {res['category']})")
    print(f"Signals: {res['signals']}")
    assert res["result"] == "REAL", f"Expected REAL, got {res['result']}"
    print("✅ Real mobile camera test passed!\n")

def test_ai_portrait_synthetic_signals():
    print("2. Testing AI Portrait Simulation (No Camera EXIF + Studio Vignette + Spectral Anomaly)...")
    # Simulate an AI generated portrait with smooth studio vignette background & latent VAE smoothness
    img_ai = np.zeros((300, 300, 3), dtype=np.uint8)
    
    # Radial studio vignette background
    cy, cx = 150, 150
    for y in range(300):
        for x in range(300):
            r = np.sqrt((x - cx)**2 + (y - cy)**2)
            val = max(50, min(230, int(220 - 0.002 * (r**2))))
            img_ai[y, x] = [val - 10, val, val + 15]
            
    # Add high-frequency grid artifacts
    for y in range(0, 300, 4):
        for x in range(0, 300, 4):
            img_ai[y:y+2, x:x+2] = np.clip(img_ai[y:y+2, x:x+2].astype(int) + 20, 0, 255)

    res = detect_deepfake_and_ai(img_ai)
    print(f"Result for AI Portrait: {res['result']} (Confidence: {res['confidence']}%, Category: {res['category']})")
    print(f"Signals: {res['signals']}")
    print(f"Metrics: {res['metrics']}")
    assert res["result"] == "FAKE", f"Expected FAKE, got {res['result']}"
    print("✅ AI Portrait test passed!\n")

def test_ai_generator_metadata():
    print("3. Testing AI Generator Prompt/Metadata Detection...")
    img = Image.new("RGB", (256, 256), color=(140, 110, 95))
    raw_ai_bytes = b"header...parameters: prompt: a photorealistic portrait of an Indian girl, Steps: 30, Sampler: DPM++ 2M, Seed: 123456, Model: SDXL_v1.0"
    res = detect_deepfake_and_ai(img, raw_bytes=raw_ai_bytes)
    print(f"Result for AI metadata image: {res['result']} (Confidence: {res['confidence']}%, Category: {res['category']})")
    assert res["result"] == "FAKE", f"Expected FAKE, got {res['result']}"
    print("✅ AI metadata test passed!\n")

if __name__ == "__main__":
    print("=" * 60)
    print("Running FakeShield Multi-Signal Detection Tests")
    print("=" * 60)
    test_real_camera_signals()
    test_ai_portrait_synthetic_signals()
    test_ai_generator_metadata()
    print("=" * 60)
    print("All tests passed successfully! 🚀")
