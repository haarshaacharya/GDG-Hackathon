from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from app.services.deepfake_detector import predict_face


app = FastAPI(
    title="FakeShield API",
    description="AI-powered deepfake detection backend",
    version="1.0.0"
)


# --------------------------------------------------
# CORS
# --------------------------------------------------

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


# --------------------------------------------------
# Face Detection Model
# --------------------------------------------------

MODEL_PATH = "models/face_detector.tflite"


# --------------------------------------------------
# Root
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "FakeShield API is running",
        "status": "online"
    }


# --------------------------------------------------
# Health Check
# --------------------------------------------------

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "FakeShield Backend"
    }


# --------------------------------------------------
# Image Prediction
# --------------------------------------------------

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):

    # --------------------------------------------------
    # Validate uploaded file
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Read uploaded image
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Decode image
    # --------------------------------------------------

    frame = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if frame is None:
        raise HTTPException(
            status_code=400,
            detail="Could not read the uploaded image."
        )

    frame_height, frame_width = frame.shape[:2]

    # --------------------------------------------------
    # Create MediaPipe face detector
    # --------------------------------------------------

    base_options = python.BaseOptions(
        model_asset_path=MODEL_PATH
    )

    options = vision.FaceDetectorOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,

        # Slightly lower threshold so smaller/
        # secondary faces have a better chance of
        # being detected.
        min_detection_confidence=0.35
    )

    detector = vision.FaceDetector.create_from_options(
        options
    )

    try:

        # --------------------------------------------------
        # BGR -> RGB
        # --------------------------------------------------

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # --------------------------------------------------
        # Detect ALL available faces
        # --------------------------------------------------

        result = detector.detect(mp_image)

        detections = result.detections or []

        # --------------------------------------------------
        # No face found
        # --------------------------------------------------

        if len(detections) == 0:
            return {
                "success": True,
                "faces_detected": 0,
                "predictions": [],
                "message": "No face detected."
            }

        predictions = []

        # --------------------------------------------------
        # Process every detected face
        # --------------------------------------------------

        for face_index, detection in enumerate(
            detections,
            start=1
        ):

            bbox = detection.bounding_box

            # --------------------------------------------------
            # Bounding box
            # --------------------------------------------------

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

            # --------------------------------------------------
            # Validate bounding box
            # --------------------------------------------------

            if x2 <= x1 or y2 <= y1:
                continue

            face_width = x2 - x1
            face_height = y2 - y1

            # Ignore extremely tiny detections
            if face_width < 20 or face_height < 20:
                continue

            # --------------------------------------------------
            # Crop face
            # --------------------------------------------------

            face_crop = frame[
                y1:y2,
                x1:x2
            ]

            if face_crop.size == 0:
                continue

            # --------------------------------------------------
            # Resize face for AI model
            # --------------------------------------------------

            try:

                face_input = cv2.resize(
                    face_crop,
                    (224, 224),
                    interpolation=cv2.INTER_AREA
                )

            except Exception:
                continue

            # --------------------------------------------------
            # Deepfake prediction
            # --------------------------------------------------

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

            # --------------------------------------------------
            # Store prediction
            # --------------------------------------------------

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

        # --------------------------------------------------
        # No usable faces
        # --------------------------------------------------

        if not predictions:
            return {
                "success": True,
                "faces_detected": 0,
                "predictions": [],
                "message": (
                    "Faces were detected, but none could "
                    "be processed."
                )
            }

        # --------------------------------------------------
        # Final response
        # --------------------------------------------------

        return {
            "success": True,
            "faces_detected": len(predictions),
            "predictions": predictions,
            "message": (
                f"{len(predictions)} face(s) analyzed successfully."
            )
        }

    finally:

        detector.close()