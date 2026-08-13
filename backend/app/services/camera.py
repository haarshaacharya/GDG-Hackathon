import cv2
import time


def start_camera():
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("❌ Camera open nahi ho raha.")
        return

    print("✅ Camera successfully connected.")
    print("Press Q to close the camera.")

    previous_time = time.time()

    while True:
        success, frame = camera.read()

        if not success:
            print("❌ Frame read nahi ho raha.")
            break

        current_time = time.time()

        elapsed_time = current_time - previous_time

        if elapsed_time > 0:
            fps = 1 / elapsed_time
        else:
            fps = 0

        previous_time = current_time

        # FPS screen par show karo
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("FakeShield - Camera Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    start_camera()