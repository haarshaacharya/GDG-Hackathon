import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification


MODEL_NAME = "prithivMLmods/deepfake-detector-model-v1"


print("Loading FakeShield AI model...")

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)

model.eval()

print("✅ Model loaded successfully.")


def predict_face(face_crop):
    """
    Predict whether an OpenCV face crop is REAL or FAKE.

    face_crop:
        OpenCV image in BGR format.
    """

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
    print("Model is ready.")