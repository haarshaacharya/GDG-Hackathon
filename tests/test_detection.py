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
    analyze_sensor_noise_and_texture,
    detect_deepfake_and_ai,
)

def test_real_camera_signals():
    print("Testing Real Mobile Camera simulation...")
    # Simulate a real camera image with natural noise & optical gradient
    np.random.seed(42)
    base = np.zeros((256, 256, 3), dtype=np.uint8)
    for y in range(256):
        for x in range(256):
            base[y, x] = [120 + int(30 * np.sin(x / 20.0)), 100 + int(20 * np.cos(y / 20.0)), 90]
    
    # Add natural camera sensor noise (Poisson-Gaussian)
    noise = np.random.normal(0, 4.5, (256, 256, 3))
    real_img = np.clip(base.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    pil_real = Image.fromarray(real_img)
    # Add fake EXIF indicating a camera phone
    res = detect_deepfake_and_ai(pil_real)
    print(f"Result for natural sensor image: {res['result']} (Confidence: {res['confidence']}%, Category: {res['category']})")
    print(f"Signals: {res['signals']}")
    assert res["result"] == "REAL", f"Expected REAL, got {res['result']}"
    print("✅ Real mobile camera test passed!\n")

def test_ai_generator_metadata():
    print("Testing AI Generator Prompt/Metadata Detection...")
    img = Image.new("RGB", (256, 256), color=(140, 110, 95))
    # Inject AI prompt info
    raw_ai_bytes = b"header...parameters: prompt: a photorealistic portrait of a human, Steps: 30, Sampler: DPM++ 2M, Seed: 123456, Model: SDXL_v1.0"
    res = detect_deepfake_and_ai(img, raw_bytes=raw_ai_bytes)
    print(f"Result for AI metadata image: {res['result']} (Confidence: {res['confidence']}%, Category: {res['category']})")
    print(f"Signals: {res['signals']}")
    assert res["result"] == "FAKE", f"Expected FAKE, got {res['result']}"
    print("✅ AI metadata test passed!\n")

def test_ai_frequency_grid_artifacts():
    print("Testing AI High-Frequency Grid / Checkerboard Artifacts...")
    # Simulate neural upsampling periodic grid pattern
    grid = np.zeros((256, 256), dtype=np.uint8) + 128
    for y in range(0, 256, 4):
        for x in range(0, 256, 4):
            grid[y:y+2, x:x+2] += 25
            
    score, signals = analyze_frequency_spectrum(grid)
    print(f"AI Spectral Score: {score:.2f}, Signals: {signals}")
    assert score > 0.60, f"Expected high AI score for grid artifact, got {score}"
    print("✅ Frequency domain AI detector test passed!\n")

if __name__ == "__main__":
    print("=" * 60)
    print("Running FakeShield Multi-Signal Detection Tests")
    print("=" * 60)
    test_real_camera_signals()
    test_ai_generator_metadata()
    test_ai_frequency_grid_artifacts()
    print("=" * 60)
    print("All tests passed successfully! 🚀")
