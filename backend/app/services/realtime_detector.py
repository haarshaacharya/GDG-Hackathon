import cv2
import mediapipe as mp
import torch
import time
from collections import deque

from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = "models/face_detector.tflite"
from app.services.deepfake_detector import detect_deepfake_and_ai

def predict_face(face_crop):
    res = detect_deepfake_and_ai(face_crop)
    return res["result"], res["confidence"] / 100.0


def start_realtime_detection():

    # --------------------------------
    # MediaPipe face detector
    # --------------------------------

    base_options = python.BaseOptions(
        model_asset_path=MODEL_PATH
    )

    options = vision.FaceDetectorOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        min_detection_confidence=0.5
    )

    detector = vision.FaceDetector.create_from_options(
        options
    )

    # --------------------------------
    # Camera
    # --------------------------------

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("❌ Camera open nahi ho raha.")
        detector.close()
        return

    print("✅ FakeShield Realtime Detection started.")
    print("Press Q to close.")

    # --------------------------------
    # Performance variables
    # --------------------------------

    frame_counter = 0

    # Last AI result
    last_label = "ANALYZING"
    last_confidence = 0.0

    # --------------------------------
    # B8: Prediction stabilization
    # --------------------------------

    prediction_history = deque(maxlen=5)

    # --------------------------------
    # FPS calculation
    # --------------------------------

    previous_time = time.perf_counter()
    fps = 0.0

    while True:

        success, frame = camera.read()

        if not success:
            print("❌ Frame read nahi ho raha.")
            break

        # Count frames
        frame_counter += 1

        # --------------------------------
        # FPS calculation
        # --------------------------------

        current_time = time.perf_counter()

        elapsed = current_time - previous_time

        if elapsed > 0:

            current_fps = 1 / elapsed

            # Smooth FPS
            fps = (fps * 0.9) + (current_fps * 0.1)

        previous_time = current_time

        # --------------------------------
        # BGR → RGB
        # --------------------------------

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # --------------------------------
        # Detect faces
        # --------------------------------

        result = detector.detect(mp_image)

        face_count = 0

        if result.detections:

            face_count = len(result.detections)

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

                x2 = min(
                    frame_width,
                    x + width
                )

                y2 = min(
                    frame_height,
                    y + height
                )

                # --------------------------------
                # Crop face
                # --------------------------------

                face_crop = frame[
                    y1:y2,
                    x1:x2
                ]

                if face_crop.size > 0:

                    # --------------------------------
                    # Resize to 224 × 224
                    # --------------------------------

                    face_input = cv2.resize(
                        face_crop,
                        (224, 224)
                    )

                    # --------------------------------
                    # AI prediction
                    # Every 10th frame only
                    # --------------------------------

                    if frame_counter % 10 == 0:

                        label, confidence = predict_face(
                            face_input
                        )

                        # Store prediction
                        prediction_history.append(
                            (label, confidence)
                        )

                        # --------------------------------
                        # Majority voting
                        # --------------------------------

                        fake_count = sum(
                            1
                            for item in prediction_history
                            if item[0] == "FAKE"
                        )

                        real_count = sum(
                            1
                            for item in prediction_history
                            if item[0] == "REAL"
                        )

                        if fake_count > real_count:
                            last_label = "FAKE"
                        else:
                            last_label = "REAL"

                        # --------------------------------
                        # Average confidence
                        # --------------------------------

                        last_confidence = sum(
                            item[1]
                            for item in prediction_history
                        ) / len(prediction_history)

                    # --------------------------------
                    # B9: Result color
                    # --------------------------------

                    if last_label == "FAKE":

                        result_color = (
                            0,
                            0,
                            255
                        )

                    elif last_label == "REAL":

                        result_color = (
                            0,
                            255,
                            0
                        )

                    else:

                        result_color = (
                            0,
                            255,
                            255
                        )

                    # --------------------------------
                    # Draw face box
                    # --------------------------------

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        result_color,
                        2
                    )

                    # --------------------------------
                    # AI result
                    # --------------------------------

                    confidence_percent = (
                        last_confidence * 100
                    )

                    text = (
                        f"{last_label} "
                        f"{confidence_percent:.1f}%"
                    )

                    cv2.putText(
                        frame,
                        text,
                        (
                            x1,
                            max(y1 - 10, 25)
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        result_color,
                        2
                    )

        # --------------------------------
        # Display FPS
        # --------------------------------

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        # --------------------------------
        # Display face count
        # --------------------------------

        cv2.putText(
            frame,
            f"Faces: {face_count}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # --------------------------------
        # Display system status
        # --------------------------------

        cv2.putText(
            frame,
            "FakeShield AI Active",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # --------------------------------
        # Display camera
        # --------------------------------

        cv2.imshow(
            "FakeShield - Realtime Deepfake Detection",
            frame
        )

        # --------------------------------
        # Q = quit
        # --------------------------------

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # --------------------------------
    # Cleanup
    # --------------------------------

    camera.release()
    detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    start_realtime_detection()