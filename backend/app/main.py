from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import cv2
import numpy as np
import mediapipe as mp
import threading

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from app.services.deepfake_detector import predict_face


app = FastAPI(
    title="FakeShield API",
    description="AI-powered deepfake detection backend",
    version="1.0.0"
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# FACE DETECTION MODEL
# =========================================================

MODEL_PATH = "models/face_detector.tflite"


# =========================================================
# Shared MediaPipe detector
# =========================================================

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

face_detector_options = vision.FaceDetectorOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    min_detection_confidence=0.35
)

face_detector = vision.FaceDetector.create_from_options(
    face_detector_options
)

# Prevent simultaneous access to the MediaPipe detector
detector_lock = threading.Lock()


# =========================================================
# Helper: Analyze frame
# =========================================================

def analyze_frame(frame):
    """
    Detect all faces in an OpenCV BGR frame
    and run deepfake prediction on every usable face.
    """

    frame_height, frame_width = frame.shape[:2]

    # BGR -> RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # MediaPipe detection
    with detector_lock:
        result = face_detector.detect(mp_image)

    detections = result.detections or []

    predictions = []

    for face_index, detection in enumerate(
        detections,
        start=1
    ):

        bbox = detection.bounding_box

        # -------------------------------------------------
        # Bounding box
        # -------------------------------------------------

        x1 = max(
            0,
            int(bbox.origin_x)
        )

        y1 = max(
            0,
            int(bbox.origin_y)
        )

        x2 = min(
            frame_width,
            int(bbox.origin_x + bbox.width)
        )

        y2 = min(
            frame_height,
            int(bbox.origin_y + bbox.height)
        )

        # Invalid box
        if x2 <= x1 or y2 <= y1:
            continue

        face_width = x2 - x1
        face_height = y2 - y1

        # Ignore tiny detections
        if face_width < 20 or face_height < 20:
            continue

        # -------------------------------------------------
        # Crop face
        # -------------------------------------------------

        face_crop = frame[
            y1:y2,
            x1:x2
        ]

        if face_crop.size == 0:
            continue

        # -------------------------------------------------
        # Resize
        # -------------------------------------------------

        try:

            face_input = cv2.resize(
                face_crop,
                (224, 224),
                interpolation=cv2.INTER_AREA
            )

        except Exception as resize_error:

            print(
                f"Face {face_index} resize failed: "
                f"{resize_error}"
            )

            continue

        # -------------------------------------------------
        # Deepfake prediction
        # -------------------------------------------------

        try:

            label, confidence = predict_face(
                face_input
            )

        except Exception as prediction_error:

            print(
                f"Face {face_index} prediction failed: "
                f"{prediction_error}"
            )

            continue

        # -------------------------------------------------
        # Store prediction
        # -------------------------------------------------

        predictions.append({
            "face": face_index,
            "result": label,
            "confidence": round(
                float(confidence) * 100,
                2
            ),
            "bounding_box": {
                "x": x1,
                "y": y1,
                "width": face_width,
                "height": face_height
            }
        })

    return predictions


# =========================================================
# Root
# =========================================================

@app.get("/")
def root():

    return {
        "message": "FakeShield API is running",
        "status": "online"
    }


# =========================================================
# Health Check
# =========================================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "service": "FakeShield Backend"
    }


# =========================================================
# IMAGE PREDICTION
# =========================================================

@app.post("/predict")
async def predict_image(
    file: UploadFile = File(...)
):

    # -----------------------------------------------------
    # Validate file
    # -----------------------------------------------------

    if not file.content_type:

        raise HTTPException(
            status_code=400,
            detail="File type could not be detected."
        )

    if not file.content_type.startswith("image/"):

        raise HTTPException(
            status_code=400,
            detail="Please upload an image file."
        )

    # -----------------------------------------------------
    # Read image
    # -----------------------------------------------------

    image_bytes = await file.read()

    if not image_bytes:

        raise HTTPException(
            status_code=400,
            detail="The uploaded image is empty."
        )

    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )

    # -----------------------------------------------------
    # Decode image
    # -----------------------------------------------------

    frame = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if frame is None:

        raise HTTPException(
            status_code=400,
            detail="Could not read the uploaded image."
        )

    # -----------------------------------------------------
    # Analyze
    # -----------------------------------------------------

    predictions = analyze_frame(frame)

    # -----------------------------------------------------
    # No usable faces
    # -----------------------------------------------------

    if not predictions:

        return {
            "success": True,
            "faces_detected": 0,
            "predictions": [],
            "message": "No face detected or no face could be processed."
        }

    # -----------------------------------------------------
    # Final response
    # -----------------------------------------------------

    return {
        "success": True,
        "faces_detected": len(predictions),
        "predictions": predictions,
        "message": (
            f"{len(predictions)} face(s) analyzed successfully."
        )
    }


# =========================================================
# LIVE CAMERA FRAME PREDICTION
# =========================================================

@app.post("/predict-frame")
async def predict_camera_frame(
    file: UploadFile = File(...)
):

    # -----------------------------------------------------
    # Validate camera frame
    # -----------------------------------------------------

    if not file.content_type:

        raise HTTPException(
            status_code=400,
            detail="Camera frame type could not be detected."
        )

    if not file.content_type.startswith("image/"):

        raise HTTPException(
            status_code=400,
            detail="Camera frame must be an image."
        )

    # -----------------------------------------------------
    # Read frame
    # -----------------------------------------------------

    image_bytes = await file.read()

    if not image_bytes:

        raise HTTPException(
            status_code=400,
            detail="Camera frame is empty."
        )

    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )

    # -----------------------------------------------------
    # Decode frame
    # -----------------------------------------------------

    frame = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if frame is None:

        raise HTTPException(
            status_code=400,
            detail="Could not decode camera frame."
        )

    # -----------------------------------------------------
    # Analyze frame
    # -----------------------------------------------------

    predictions = analyze_frame(frame)

    # -----------------------------------------------------
    # No face
    # -----------------------------------------------------

    if not predictions:

        return {
            "success": True,
            "faces_detected": 0,
            "predictions": [],
            "message": "No face detected."
        }

    # -----------------------------------------------------
    # Camera response
    # -----------------------------------------------------

    return {
        "success": True,
        "faces_detected": len(predictions),
        "predictions": predictions
    }