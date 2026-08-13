import cv2
import mediapipe as mp
import torch

from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = "models/face_detector.tflite"
AI_MODEL = "prithivMLmods/deepfake-detector-model-v1"


print("Loading FakeShield AI model...")

processor = AutoImageProcessor.from_pretrained(AI_MODEL)
model = AutoModelForImageClassification.from_pretrained(AI_MODEL)

model.eval()

print("✅ Deepfake AI model loaded.")


def predict_face(face_crop):

    # OpenCV BGR → RGB
    rgb_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

    # NumPy → PIL
    image = Image.fromarray(rgb_face)

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


def start_realtime_detection():

    # MediaPipe face detector
    base_options = python.BaseOptions(
        model_asset_path=MODEL_PATH
    )

    options = vision.FaceDetectorOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        min_detection_confidence=0.5
    )

    detector = vision.FaceDetector.create_from_options(options)

    # Camera
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("❌ Camera open nahi ho raha.")
        detector.close()
        return

    print("✅ FakeShield Realtime Detection started.")
    print("Press Q to close.")

    while True:

        success, frame = camera.read()

        if not success:
            print("❌ Frame read nahi ho raha.")
            break

        # BGR → RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Detect faces
        result = detector.detect(mp_image)

        if result.detections:

            for detection in result.detections:

                bbox = detection.bounding_box

                x = bbox.origin_x
                y = bbox.origin_y
                width = bbox.width
                height = bbox.height

                # Keep coordinates inside frame
                frame_height, frame_width = frame.shape[:2]

                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(frame_width, x + width)
                y2 = min(frame_height, y + height)

                # Crop face
                face_crop = frame[y1:y2, x1:x2]

                if face_crop.size > 0:

                    # Resize to 224 × 224
                    face_input = cv2.resize(
                        face_crop,
                        (224, 224)
                    )

                    # Deepfake prediction
                    label, confidence = predict_face(
                        face_input
                    )

                    confidence_percent = confidence * 100

                    # Draw bounding box
                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    # Display result
                    text = (
                        f"{label} "
                        f"{confidence_percent:.1f}%"
                    )

                    cv2.putText(
                        frame,
                        text,
                        (x1, max(y1 - 10, 25)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )

        # Show camera
        cv2.imshow(
            "FakeShield - Realtime Deepfake Detection",
            frame
        )

        # Q = quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    start_realtime_detection()