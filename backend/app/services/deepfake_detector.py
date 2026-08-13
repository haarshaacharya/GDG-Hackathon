import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification


MODEL_NAME = "prithivMLmods/deepfake-detector-model-v1"


print("Loading FakeShield AI model...")

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)

model.eval()

print("✅ Model loaded successfully.")


def predict_image(image_path):

    image = Image.open(image_path).convert("RGB")

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