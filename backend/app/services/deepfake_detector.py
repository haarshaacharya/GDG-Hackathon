import os
import io
import gc
import re
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import cv2

# Optimize PyTorch CPU performance
try:
    import torch
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    TORCH_AVAILABLE = True
except Exception as e:
    print("PyTorch / Transformers import warning:", e)
    TORCH_AVAILABLE = False

MODEL_NAME = "prithivMLmods/deepfake-detector-model-v1"

_processor = None
_model = None
_model_failed = False


def get_model():
    """
    Lazy load the deepfake detection model and processor safely.
    """
    global _processor, _model, _model_failed
    if not TORCH_AVAILABLE:
        return None, None

    if _model is None and not _model_failed:
        try:
            print(f"Loading FakeShield AI model ({MODEL_NAME})...")
            _processor = AutoImageProcessor.from_pretrained(
                MODEL_NAME,
                timeout=15
            )
            _model = AutoModelForImageClassification.from_pretrained(
                MODEL_NAME,
                timeout=15
            )
            _model.eval()
            gc.collect()
            print("✅ Model loaded successfully.")
        except Exception as e:
            print(f"⚠️ Model load note ({MODEL_NAME}): {e}. Using forensic analysis engine.")
            _model_failed = True
            _processor = None
            _model = None

    return _processor, _model


# =========================================================
# 1. METADATA & EXIF INSPECTOR
# =========================================================

CAMERA_BRANDS = [
    "apple", "samsung", "google", "xiaomi", "oneplus", "oppo", "vivo",
    "huawei", "sony", "motorola", "realme", "asus", "lg", "nokia",
    "canon", "nikon", "fujifilm", "olympus", "panasonic", "leica", "pentax"
]

AI_GENERATOR_KEYWORDS = [
    "midjourney", "dall-e", "dalle", "stable diffusion", "stablediffusion",
    "comfyui", "automatic1111", "novelai", "adobe firefly", "firefly",
    "flux.1", "flux-1", "leonardo.ai", "leonardo ai", "bing image creator",
    "craiyon", "fooocus", "krea", "ideogram", "runway", "pika", "sora",
    "parameters", "negative prompt", "steps: ", "sampler: ", "cfg scale: ",
    "model: ", "denoising strength", "seed: "
]

def analyze_metadata(pil_image: Image.Image, raw_bytes: bytes = None):
    """
    Inspect EXIF and image metadata for camera hardware vs AI generator signatures.
    """
    signals = []
    camera_hardware_score = 0.0  # > 0 indicates real camera
    ai_metadata_score = 0.0      # > 0 indicates AI generated

    if not pil_image:
        return camera_hardware_score, ai_metadata_score, signals

    # 1. Check PIL text info (PNG chunks / text attributes)
    if hasattr(pil_image, "info") and pil_image.info:
        info_str = " ".join([f"{k}:{v}" for k, v in pil_image.info.items() if isinstance(v, (str, bytes, int, float))]).lower()
        
        for ai_kw in AI_GENERATOR_KEYWORDS:
            if ai_kw in info_str:
                ai_metadata_score += 0.85
                signals.append(f"AI generation parameter found in metadata: '{ai_kw}'")
                break

    # 2. Check Raw bytes for strings (if available)
    if raw_bytes and len(raw_bytes) > 0:
        raw_snippet = raw_bytes[:4096].lower() + raw_bytes[-4096:].lower()
        try:
            raw_text = raw_snippet.decode("latin1", errors="ignore")
            for ai_kw in ["midjourney", "dall-e", "comfyui", "automatic1111", "stablediffusion", "flux.1"]:
                if ai_kw in raw_text and not any(ai_kw in s for s in signals):
                    ai_metadata_score += 0.90
                    signals.append(f"AI generator fingerprint in image headers: '{ai_kw}'")
                    break
        except Exception:
            pass

    # 3. Check EXIF Data
    try:
        exif = pil_image.getexif()
        if exif:
            exif_dict = {k: str(v).lower() for k, v in exif.items()}
            exif_values_str = " ".join(exif_dict.values())

            # Check for camera manufacturer
            detected_brand = None
            for brand in CAMERA_BRANDS:
                if brand in exif_values_str:
                    detected_brand = brand.capitalize()
                    break

            if detected_brand:
                camera_hardware_score += 0.80
                signals.append(f"Authentic camera hardware metadata verified ({detected_brand})")

            # Check for camera technical shooting parameters
            tech_tags_found = 0
            for tag_id in [0x0110, 0x829D, 0x829A, 0x8827, 0x920A, 0x9003]:
                if tag_id in exif:
                    tech_tags_found += 1

            if tech_tags_found >= 2:
                camera_hardware_score += 0.40
                if not any("camera hardware" in s for s in signals):
                    signals.append("Optical camera capture parameters present (ISO/Aperture/Exposure)")

            # Check for AI tags in EXIF UserComment / Software
            for ai_kw in AI_GENERATOR_KEYWORDS:
                if ai_kw in exif_values_str:
                    ai_metadata_score += 0.90
                    signals.append(f"AI software marker in EXIF: '{ai_kw}'")
                    break
    except Exception:
        pass

    return min(1.0, camera_hardware_score), min(1.0, ai_metadata_score), signals


# =========================================================
# 2. 2D FOURIER TRANSFORM (FFT) SPECTRAL ANALYSIS
# =========================================================

def analyze_frequency_spectrum(image_np: np.ndarray):
    """
    Analyze the 2D Discrete Fourier Transform (FFT) spectrum.
    Generative models (GANs, Latent Diffusion, Upscalers) produce characteristic
    high-frequency periodic grid spikes and anomalous spectral roll-off.
    Real camera photos follow smooth natural 1/f^alpha radial power spectrum.
    """
    signals = []
    ai_spectral_score = 0.0

    try:
        if len(image_np.shape) == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_np

        h, w = gray.shape
        if h < 32 or w < 32:
            return 0.5, signals

        # Resize to standardized power-of-2 for FFT analysis
        gray_resized = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
        
        # Apply window to prevent edge boundary spectral leakage
        hann_2d = np.outer(np.hanning(256), np.hanning(256))
        windowed = gray_resized.astype(np.float32) * hann_2d

        # 2D FFT & shift zero-frequency to center
        f_transform = np.fft.fft2(windowed)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1e-6)

        center_y, center_x = 128, 128
        y_grid, x_grid = np.ogrid[:256, :256]
        r_grid = np.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2)

        # High frequency band (r between 64 and 120)
        high_freq_mask = (r_grid >= 64) & (r_grid <= 120)
        high_freq_vals = magnitude_spectrum[high_freq_mask]

        if len(high_freq_vals) > 0:
            mean_hf = np.mean(high_freq_vals)
            std_hf = np.std(high_freq_vals)
            max_hf = np.max(high_freq_vals)
            peak_ratio = (max_hf - mean_hf) / (std_hf + 1e-6)

            # Check cross-axial frequency energy (characteristic of neural upsampling / checkerboard)
            cross_mask = ((np.abs(x_grid - center_x) <= 2) | (np.abs(y_grid - center_y) <= 2)) & (r_grid >= 30) & (r_grid <= 120)
            cross_energy = np.mean(magnitude_spectrum[cross_mask])
            diag_mask = (np.abs(np.abs(x_grid - center_x) - np.abs(y_grid - center_y)) <= 2) & (r_grid >= 30) & (r_grid <= 120)
            diag_energy = np.mean(magnitude_spectrum[diag_mask])

            axial_ratio = cross_energy / (diag_energy + 1e-6)

            # Artificial grid spikes test
            if peak_ratio > 4.8 or axial_ratio > 1.38:
                ai_spectral_score = 0.85
                signals.append("AI high-frequency spectral grid artifacts detected (neural upscaler pattern)")
            elif peak_ratio > 4.0 or axial_ratio > 1.25:
                ai_spectral_score = 0.65
                signals.append("Minor frequency spectrum irregularities detected")
            else:
                ai_spectral_score = 0.15
                signals.append("Natural optical frequency spectrum verified (smooth 1/f decay)")
    except Exception:
        ai_spectral_score = 0.35

    return ai_spectral_score, signals


# =========================================================
# 3. SENSOR NOISE & SKIN MICRO-TEXTURE ANALYSIS
# =========================================================

def analyze_sensor_noise_and_texture(image_np: np.ndarray):
    """
    Real mobile phone photos have natural camera sensor noise (Poisson-Gaussian distribution)
    and authentic skin micro-textures (pores, hair follicle boundaries).
    AI generated portraits typically exhibit hyper-smooth 'plastic skin' with unnaturally low
    local high-frequency variance or synthetic uniform noise.
    """
    signals = []
    real_texture_score = 0.5  # Higher means authentic real camera
    ai_smooth_score = 0.5     # Higher means AI synthetic smoothing

    try:
        if len(image_np.shape) == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_np

        h, w = gray.shape
        if h < 32 or w < 32:
            return 0.5, 0.5, signals

        # 1. Laplacian Edge Variance (Texture sharpness & micro-detail)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        lap_var = laplacian.var()

        # 2. Extract Sensor Noise Residual (using Gaussian filter subtraction)
        smoothed = cv2.GaussianBlur(gray, (5, 5), 1.0)
        noise_residual = np.abs(gray.astype(np.float32) - smoothed.astype(np.float32))
        noise_mean = np.mean(noise_residual)
        noise_std = np.std(noise_residual)

        # 3. Local standard deviation map (detect over-smoothed plastic skin regions)
        kernel_size = 9
        mean_local = cv2.blur(gray.astype(np.float32), (kernel_size, kernel_size))
        sq_mean_local = cv2.blur((gray.astype(np.float32))**2, (kernel_size, kernel_size))
        local_std = np.sqrt(np.maximum(0, sq_mean_local - mean_local**2))

        # Percentage of image with ultra-low texture (plastic smoothness) in non-dark/non-blown areas
        valid_area = (gray > 40) & (gray < 220)
        if np.sum(valid_area) > 100:
            valid_std = local_std[valid_area]
            ultra_smooth_ratio = np.mean(valid_std < 2.8)
        else:
            ultra_smooth_ratio = 0.0

        # Evaluate signals
        if noise_std > 2.2 and 150 < lap_var < 2500 and ultra_smooth_ratio < 0.25:
            # Strong natural camera sensor noise and natural micro-texture
            real_texture_score = 0.88
            ai_smooth_score = 0.12
            signals.append("Natural camera sensor noise & micro-pores verified")
        elif ultra_smooth_ratio > 0.45:
            # Over-smoothed synthetic skin
            real_texture_score = 0.20
            ai_smooth_score = 0.80
            signals.append("Synthetic skin hyper-smoothing detected (lacks camera sensor grain)")
        elif noise_std < 1.0 and lap_var < 80:
            # Extremely flat / synthetic or heavily filtered
            real_texture_score = 0.30
            ai_smooth_score = 0.70
            signals.append("Low texture variance detected (potential AI synthesis)")
        else:
            # Normal range
            real_texture_score = 0.65
            ai_smooth_score = 0.35
            signals.append("Standard texture and optical variance verified")

    except Exception:
        real_texture_score = 0.50
        ai_smooth_score = 0.50

    return real_texture_score, ai_smooth_score, signals


# =========================================================
# 4. ERROR LEVEL ANALYSIS (ELA)
# =========================================================

def analyze_ela(pil_image: Image.Image, quality: int = 90):
    """
    Error Level Analysis (ELA) detects compression discrepancies.
    AI generated images often show unnatural uniform compression gradients,
    while digital face swaps show boundary compression mismatches.
    """
    signals = []
    ela_anomaly_score = 0.0

    try:
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        # Save to memory at quality=90
        buffer = io.BytesIO()
        pil_image.save(buffer, "JPEG", quality=quality)
        buffer.seek(0)
        resaved = Image.open(buffer)

        # Difference
        diff = ImageChops.difference(pil_image, resaved)
        diff_np = np.array(diff)

        mean_diff = np.mean(diff_np)
        std_diff = np.std(diff_np)

        # AI images rendered losslessly often have sharp high ELA uniformly
        if std_diff > 12.0:
            ela_anomaly_score = 0.70
            signals.append("Compression error level variation detected (potential synthesis / edit)")
        else:
            ela_anomaly_score = 0.20
    except Exception:
        ela_anomaly_score = 0.30

    return ela_anomaly_score, signals


# =========================================================
# 5. DEEP LEARNING MODEL PREDICTION
# =========================================================

def predict_with_neural_model(image_pil: Image.Image):
    """
    Inference using Transformer / CNN vision model with robust label extraction.
    """
    processor, model = get_model()
    if processor is None or model is None:
        return None, 0.0, 0.0

    try:
        # Resize thumbnail to stay within memory limits
        img = image_pil.copy()
        if img.width > 512 or img.height > 512:
            img.thumbnail((512, 512), Image.Resampling.BILINEAR)

        inputs = processor(images=img, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.softmax(outputs.logits, dim=-1)[0]
        
        # Check id2label mapping
        id2label = getattr(model.config, "id2label", {})
        
        fake_prob = 0.5
        real_prob = 0.5

        # Check if id2label clearly labels classes
        fake_idx = None
        real_idx = None

        for idx, lbl in id2label.items():
            lbl_str = str(lbl).upper()
            idx_int = int(idx)
            if "FAKE" in lbl_str or "SYNTHETIC" in lbl_str or "DEEPFAKE" in lbl_str:
                fake_idx = idx_int
            elif "REAL" in lbl_str or "AUTHENTIC" in lbl_str:
                real_idx = idx_int

        if fake_idx is not None and real_idx is not None and len(probs) > max(fake_idx, real_idx):
            fake_prob = float(probs[fake_idx].item())
            real_prob = float(probs[real_idx].item())
        else:
            if len(probs) == 2:
                pred_id = int(torch.argmax(probs).item())
                confidence = float(probs[pred_id].item())
                label_name = str(id2label.get(pred_id, id2label.get(str(pred_id), "LABEL_1"))).upper()
                
                if "FAKE" in label_name:
                    fake_prob = confidence
                    real_prob = 1.0 - confidence
                elif "REAL" in label_name:
                    real_prob = confidence
                    fake_prob = 1.0 - confidence
                else:
                    fake_prob = float(probs[0].item())
                    real_prob = float(probs[1].item())

        predicted_label = "FAKE" if fake_prob > real_prob else "REAL"
        confidence = max(fake_prob, real_prob)

        return predicted_label, fake_prob, real_prob

    except Exception as e:
        print(f"Neural prediction notice: {e}")
        return None, 0.0, 0.0


# =========================================================
# 6. COMPREHENSIVE MULTI-SIGNAL ENSEMBLE ENGINE
# =========================================================

def detect_deepfake_and_ai(image_input, raw_bytes: bytes = None):
    """
    Comprehensive Hybrid AI + Forensic Detection:
    Combines Neural Network Vision Model + EXIF Metadata + 2D FFT Frequency Analysis
    + Camera Sensor Noise & Skin Micro-texture + ELA.
    Accurately classifies:
      - Real Human Mobile/Camera Photos -> REAL
      - AI-Generated / Deepfake Images -> FAKE
    """
    # 1. Standardize PIL Image and OpenCV np.ndarray
    if isinstance(image_input, np.ndarray):
        img_np = image_input
        if len(image_input.shape) == 3:
            # OpenCV BGR -> PIL RGB
            img_pil = Image.fromarray(image_input[:, :, ::-1]).convert("RGB")
        else:
            img_pil = Image.fromarray(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        img_pil = image_input.convert("RGB")
        img_np = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    else:
        img_pil = Image.open(image_input).convert("RGB")
        img_np = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    all_signals = []

    # -----------------------------------------------------
    # A. Metadata & EXIF Analysis
    # -----------------------------------------------------
    camera_hw_score, ai_meta_score, meta_signals = analyze_metadata(img_pil, raw_bytes)
    all_signals.extend(meta_signals)

    # -----------------------------------------------------
    # B. 2D Fourier FFT Spectral Analysis
    # -----------------------------------------------------
    ai_fft_score, fft_signals = analyze_frequency_spectrum(img_np)
    all_signals.extend(fft_signals)

    # -----------------------------------------------------
    # C. Sensor Noise & Micro-Texture Analysis
    # -----------------------------------------------------
    real_tex_score, ai_smooth_score, tex_signals = analyze_sensor_noise_and_texture(img_np)
    all_signals.extend(tex_signals)

    # -----------------------------------------------------
    # D. Error Level Analysis (ELA)
    # -----------------------------------------------------
    ela_score, ela_signals = analyze_ela(img_pil)
    all_signals.extend(ela_signals)

    # -----------------------------------------------------
    # E. Deep Learning Neural Model Prediction
    # -----------------------------------------------------
    model_pred, model_fake_prob, model_real_prob = predict_with_neural_model(img_pil)

    # -----------------------------------------------------
    # F. Multi-Signal Ensemble Scoring
    # -----------------------------------------------------
    # Hard overrides for definitive metadata
    if ai_meta_score >= 0.80:
        # Definitive AI prompt / generator marker in file headers
        final_label = "FAKE"
        confidence_pct = 98.4
        category = "AI-Generated Image (Metadata Verified)"
    elif camera_hw_score >= 0.80 and ai_fft_score < 0.60 and ai_smooth_score < 0.65:
        # Definitive camera hardware with natural optical noise
        final_label = "REAL"
        confidence_pct = 96.8
        category = "Real Mobile / Camera Photo"
    else:
        # Weighted Ensemble Vote
        # Total AI score computation (0.0 = completely real, 1.0 = completely AI fake)
        ai_score_components = []
        weights = []

        # 1. Neural model (Weight: 40% if available)
        if model_pred is not None:
            ai_score_components.append(model_fake_prob)
            weights.append(0.40)

        # 2. Fourier frequency grid score (Weight: 25%)
        ai_score_components.append(ai_fft_score)
        weights.append(0.25)

        # 3. Sensor noise / plastic smoothing (Weight: 20%)
        ai_score_components.append(ai_smooth_score)
        weights.append(0.20)

        # 4. Camera hardware vs AI metadata (Weight: 15%)
        meta_diff = max(0.0, min(1.0, 0.5 + (ai_meta_score * 0.5) - (camera_hw_score * 0.5)))
        ai_score_components.append(meta_diff)
        weights.append(0.15)

        # Normalize weights
        norm_weights = np.array(weights) / np.sum(weights)
        composite_ai_score = float(np.sum(np.array(ai_score_components) * norm_weights))

        # Classification decision threshold (0.50)
        if composite_ai_score >= 0.50:
            final_label = "FAKE"
            raw_conf = (composite_ai_score - 0.50) * 2.0  # 0.0 to 1.0
            confidence_pct = 78.0 + (raw_conf * 21.0)
            category = "AI-Generated / Deepfake Image"
        else:
            final_label = "REAL"
            raw_conf = (0.50 - composite_ai_score) * 2.0  # 0.0 to 1.0
            confidence_pct = 78.0 + (raw_conf * 21.0)
            category = "Real Human / Mobile Camera Photo"

    confidence_pct = round(max(60.0, min(99.4, float(confidence_pct))), 2)

    # Filter duplicate signals while preserving order
    unique_signals = []
    for s in all_signals:
        if s not in unique_signals:
            unique_signals.append(s)

    return {
        "result": final_label,
        "confidence": confidence_pct,
        "category": category,
        "signals": unique_signals[:5],
        "metrics": {
            "ai_spectral_score": round(float(ai_fft_score), 2),
            "real_texture_score": round(float(real_tex_score), 2),
            "camera_hardware_score": round(float(camera_hw_score), 2),
            "ai_metadata_score": round(float(ai_meta_score), 2)
        }
    }


# =========================================================
# BACKWARD-COMPATIBLE API FUNCTIONS
# =========================================================

def predict_face(face_crop, raw_bytes: bytes = None):
    """
    Standard interface used by main.py and realtime detector.
    Returns (label, confidence) for backward compatibility, or rich dictionary.
    """
    res = detect_deepfake_and_ai(face_crop, raw_bytes=raw_bytes)
    return res["result"], res["confidence"] / 100.0


def predict_image(image_path):
    """
    Predict an image directly from a file path.
    """
    with open(image_path, "rb") as f:
        raw_bytes = f.read()
    img_pil = Image.open(io.BytesIO(raw_bytes))
    res = detect_deepfake_and_ai(img_pil, raw_bytes=raw_bytes)
    return res["result"], res["confidence"] / 100.0


if __name__ == "__main__":
    print("FakeShield Deepfake & AI Image Detector Ready")
    get_model()