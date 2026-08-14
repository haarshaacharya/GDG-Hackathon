import gc
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Optimize memory footprint for 512MB RAM instances (Render Free Tier)
torch.set_num_threads(1)
torch.set_grad_enabled(False)

MODEL_NAME = "prithivMLmods/deepfake-detector-model-v1"

_processor = None
_model = None


def get_model():
    """
    Lazy load the deepfake detection model and processor with low memory usage.
    """
    global _processor, _model
    if _model is None or _processor is None:
        print("Loading FakeShield AI model (low memory mode)...")
        _processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
        _model = AutoModelForImageClassification.from_pretrained(
            MODEL_NAME,
            low_cpu_mem_usage=True
        )
        _model.eval()
        gc.collect()
        print("✅ Model loaded successfully.")
    return _processor, _model



def predict_face(face_crop):
    """
    Predict whether an OpenCV face crop or PIL Image is REAL or FAKE.
    """
    processor, model = get_model()

    if isinstance(face_crop, np.ndarray):
        image = Image.fromarray(
            face_crop[:, :, ::-1]
        ).convert("RGB")
    elif isinstance(face_crop, Image.Image):
        image = face_crop.convert("RGB")
    else:
        image = Image.fromarray(np.array(face_crop)).convert("RGB")

    # Prepare model input
    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    # AI prediction
    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1
    )[0]

    predicted_id = torch.argmax(probabilities, dim=-1).item()
    confidence = probabilities[predicted_id].item()

    if hasattr(model.config, "id2label") and model.config.id2label:
        label = model.config.id2label.get(predicted_id, "REAL" if predicted_id == 1 else "FAKE")
    else:
        fake_score = probabilities[0].item()
        real_score = probabilities[1].item()
        if fake_score > real_score:
            label, confidence = "FAKE", fake_score
        else:
            label, confidence = "REAL", real_score

    label_str = str(label).upper()
    if "FAKE" in label_str:
        clean_label = "FAKE"
    elif "REAL" in label_str:
        clean_label = "REAL"
    else:
        clean_label = label_str

    return clean_label, confidence


def predict_image(image_path):
    """
    Predict an image directly from a file path.
    """
    processor, model = get_model()

    image = Image.open(
        image_path
    ).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1
    )[0]

    fake_score = probabilities[0].item()
    real_score = probabilities[1].item()

    if fake_score > real_score:
        label = "FAKE"
        confidence = fake_score
    else:
        label = "REAL"
        confidence = real_score

    return label, confidence


if __name__ == "__main__":
    print("FakeShield Deepfake Detector")
    get_model()
    print("Model is ready.")