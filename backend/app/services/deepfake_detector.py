import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification


MODEL_NAME = "prithivMLmods/deepfake-detector-model-v1"

_processor = None
_model = None


def get_model():
    """
    Lazy load the deepfake detection model and processor.
    This allows the web server to bind to the port immediately on startup.
    """
    global _processor, _model
    if _model is None or _processor is None:
        print("Loading FakeShield AI model...")
        _processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
        _model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
        _model.eval()
        print("✅ Model loaded successfully.")
    return _processor, _model


def predict_face(face_crop):
    """
    Predict whether an OpenCV face crop is REAL or FAKE.

    face_crop:
        OpenCV image in BGR format.
    """
    processor, model = get_model()

    # OpenCV BGR → RGB
    image = Image.fromarray(
        face_crop[:, :, ::-1]
    ).convert("RGB")

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

    fake_score = probabilities[0].item()
    real_score = probabilities[1].item()

    if fake_score > real_score:
        return "FAKE", fake_score

    return "REAL", real_score


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