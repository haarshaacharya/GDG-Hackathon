import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = "models/face_detector.tflite"


def start_face_detection():
    # MediaPipe Face Detector
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

    print("✅ Face Detection started.")
    print("Press Q to close.")

    while True:
        success, frame = camera.read()

        if not success:
            print("❌ Frame read nahi ho raha.")
            break

        # OpenCV BGR → RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # MediaPipe Image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Detect faces
        result = detector.detect(mp_image)

        # Draw detected faces
        if result.detections:
            for detection in result.detections:
                bbox = detection.bounding_box

                x = bbox.origin_x
                y = bbox.origin_y
                width = bbox.width
                height = bbox.height

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + width, y + height),
                    (0, 255, 0),
                    2
                )

                score = detection.categories[0].score

                cv2.putText(
                    frame,
                    f"Face: {score * 100:.1f}%",
                    (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        cv2.imshow("FakeShield - Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    start_face_detection()