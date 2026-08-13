import cv2
import mediapipe as mp
import time
from collections import deque

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = "models/face_detector.tflite"


def start_face_detection():

    base_options = python.BaseOptions(
        model_asset_path=MODEL_PATH
    )

    options = vision.FaceDetectorOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        min_detection_confidence=0.5
    )

    detector = vision.FaceDetector.create_from_options(options)

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("❌ Camera open nahi ho raha.")
        detector.close()
        return

    print("✅ FakeShield Face Detection started.")
    print("Press Q to close.")

    # FPS calculation
    previous_time = time.perf_counter()
    fps_history = deque(maxlen=30)

    while True:

        success, frame = camera.read()

        if not success:
            print("❌ Frame read nahi ho raha.")
            break

        # BGR → RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Face detection
        result = detector.detect(mp_image)

        # Draw faces
        face_count = 0

        if result.detections:

            face_count = len(result.detections)

            for detection in result.detections:

                bbox = detection.bounding_box

                x = bbox.origin_x
                y = bbox.origin_y
                width = bbox.width
                height = bbox.height

                # Keep coordinates inside the frame
                frame_height, frame_width = frame.shape[:2]

                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(frame_width, x + width)
                y2 = min(frame_height, y + height)

                # Crop face
            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size > 0:
               # Resize face for AI model
                face_input = cv2.resize(
               face_crop,
               (224, 224)
                )
 
                # Show processed face
                cv2.imshow(
                "FakeShield - Face Input 224x224",
                face_input
                )

                # Draw face rectangle
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

        # FPS
        current_time = time.perf_counter()
        elapsed = current_time - previous_time
        previous_time = current_time

        if elapsed > 0:
            current_fps = 1 / elapsed
            fps_history.append(current_fps)

        # Average FPS
        average_fps = (
            sum(fps_history) / len(fps_history)
            if fps_history
            else 0
        )

        # Display FPS
        cv2.putText(
            frame,
            f"FPS: {average_fps:.1f}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        # Display face count
        cv2.putText(
            frame,
            f"Faces: {face_count}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "FakeShield - Face Detection",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    start_face_detection()