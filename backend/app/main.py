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

    # Check image type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload an image file."
        )

    # Read uploaded image
    image_bytes = await file.read()

    # Convert bytes to NumPy array
    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )

    # Decode image
    frame = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if frame is None:
        raise HTTPException(
            status_code=400,
            detail="Could not read the uploaded image."
        )

    # --------------------------------------------------
    # Create MediaPipe face detector
    # --------------------------------------------------

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

    try:

        # BGR -> RGB
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

        # --------------------------------------------------
        # No face found
        # --------------------------------------------------

        if not result.detections:
            return {
                "success": True,
                "faces_detected": 0,
                "message": "No face detected."
            }

        predictions = []

        frame_height, frame_width = frame.shape[:2]

        # --------------------------------------------------
        # Process every detected face
        # --------------------------------------------------

        for detection in result.detections:

            bbox = detection.bounding_box

            x1 = max(0, bbox.origin_x)
            y1 = max(0, bbox.origin_y)

            x2 = min(
                frame_width,
                bbox.origin_x + bbox.width
            )

            y2 = min(
                frame_height,
                bbox.origin_y + bbox.height
            )

            # Crop face
            face_crop = frame[
                y1:y2,
                x1:x2
            ]

            if face_crop.size == 0:
                continue

            # Resize face for AI model
            face_input = cv2.resize(
                face_crop,
                (224, 224)
            )

            # Deepfake prediction
            label, confidence = predict_face(
                face_input
            )

            predictions.append({
                "result": label,
                "confidence": round(
                    confidence * 100,
                    2
                )
            })

        # --------------------------------------------------
        # Face processing failed
        # --------------------------------------------------

        if not predictions:
            return {
                "success": True,
                "faces_detected": 0,
                "message": "Face crop could not be processed."
            }

        # --------------------------------------------------
        # Final response
        # --------------------------------------------------

        return {
            "success": True,
            "faces_detected": len(predictions),
            "predictions": predictions
        }

    finally:
        detector.close()