import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["ABSL_LOG_MINIMUM_SEVERITY"] = "2"

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware


import io
from PIL import Image, ImageOps
import cv2
import numpy as np
import mediapipe as mp
import threading
from pathlib import Path

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from app.services.deepfake_detector import predict_face


# =========================================================
# APP
# =========================================================

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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# CONFIG
# =========================================================

# Maximum uploaded file size:
# 10 MB

MAX_FILE_SIZE = 10 * 1024 * 1024


# =========================================================
# FACE DETECTOR MODEL
# =========================================================

# Build model path relative to backend folder
# instead of depending on the terminal's current directory.

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "face_detector.tflite"


if not MODEL_PATH.exists():

    raise RuntimeError(
        f"Face detector model not found: {MODEL_PATH}"
    )


# =========================================================
# SHARED MEDIAPIPE DETECTOR
# =========================================================

base_options = python.BaseOptions(
    model_asset_path=str(MODEL_PATH)
)


face_detector_options = vision.FaceDetectorOptions(
    base_options=base_options,

    running_mode=vision.RunningMode.IMAGE,

    min_detection_confidence=0.20
)


face_detector = vision.FaceDetector.create_from_options(
    face_detector_options
)

# Secondary Haar cascade detectors as fallback
haar_cascades = []
for cascade_name in [
    "haarcascade_frontalface_default.xml",
    "haarcascade_frontalface_alt2.xml",
    "haarcascade_frontalface_alt.xml",
    "haarcascade_profileface.xml",
]:
    try:
        cascade_file = cv2.data.haarcascades + cascade_name
        if os.path.exists(cascade_file):
            haar_cascades.append(cv2.CascadeClassifier(cascade_file))
    except Exception as cascade_err:
        print(f"Error loading cascade {cascade_name}:", cascade_err)


# =========================================================
# THREAD LOCKS
# =========================================================

# MediaPipe detector is shared between requests.

detector_lock = threading.Lock()


# Deepfake model may also be shared between requests.

prediction_lock = threading.Lock()


# =========================================================
# HELPER: READ IMAGE
# =========================================================

async def read_uploaded_image(
    file: UploadFile,
    error_prefix="Image"
):
    """
    Safely read and decode an uploaded image.
    """

    # -----------------------------------------------------
    # Validate content type
    # -----------------------------------------------------

    if not file.content_type:

        raise HTTPException(
            status_code=400,
            detail=f"{error_prefix} type could not be detected."
        )


    if not file.content_type.startswith("image/"):

        raise HTTPException(
            status_code=400,
            detail=f"{error_prefix} must be an image."
        )


    # -----------------------------------------------------
    # Read file
    # -----------------------------------------------------

    try:

        image_bytes = await file.read()

    except Exception as error:

        print(
            f"{error_prefix} read error: {error}"
        )

        raise HTTPException(
            status_code=400,
            detail=f"Could not read {error_prefix.lower()}."
        )


    # -----------------------------------------------------
    # Empty file
    # -----------------------------------------------------

    if not image_bytes:

        raise HTTPException(
            status_code=400,
            detail=f"{error_prefix} is empty."
        )


    # -----------------------------------------------------
    # File size protection
    # -----------------------------------------------------

    if len(image_bytes) > MAX_FILE_SIZE:

        raise HTTPException(
            status_code=413,
            detail=(
                f"{error_prefix} is too large. "
                f"Maximum allowed size is 10 MB."
            )
        )


    # -----------------------------------------------------
    # Convert bytes -> PIL (with EXIF auto-orientation) -> OpenCV BGR
    # -----------------------------------------------------

    try:
        pil_image = Image.open(io.BytesIO(image_bytes))
        pil_image = ImageOps.exif_transpose(pil_image)

        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        frame = cv2.cvtColor(
            np.array(pil_image),
            cv2.COLOR_RGB2BGR
        )
    except Exception:
        image_array = np.frombuffer(
            image_bytes,
            dtype=np.uint8
        )

        frame = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

    if frame is None:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Could not decode the uploaded "
                f"{error_prefix.lower()}."
            )
        )

    return frame


def apply_nms(boxes, iou_threshold=0.50):
    """
    Remove overlapping duplicate bounding boxes using Non-Maximum Suppression.
    """
    if not boxes:
        return []

    sorted_boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    kept = []

    for current in sorted_boxes:
        cx1, cy1, cw, ch = current
        cx2, cy2 = cx1 + cw, cy1 + ch
        current_area = cw * ch

        keep = True
        for (kx1, ky1, kw, kh) in kept:
            kx2, ky2 = kx1 + kw, ky1 + kh

            # Intersection
            ix1 = max(cx1, kx1)
            iy1 = max(cy1, ky1)
            ix2 = min(cx2, kx2)
            iy2 = min(cy2, ky2)

            iw = max(0, ix2 - ix1)
            ih = max(0, iy2 - iy1)
            intersection = iw * ih

            if intersection > 0:
                union = current_area + (kw * kh) - intersection
                iou = intersection / union if union > 0 else 0
                overlap_ratio = intersection / current_area if current_area > 0 else 0

                if iou > iou_threshold or overlap_ratio > 0.50:
                    keep = False
                    break

        if keep:
            kept.append(current)

    return kept


# =========================================================
# HELPER: ANALYZE FRAME
# =========================================================

def analyze_frame(frame, allow_fallback=False):
    """
    Detect all faces in an OpenCV BGR frame and run deepfake prediction.
    If MediaPipe detects no faces, tries OpenCV Haar Cascade fallback.
    If still no faces detected and allow_fallback is True, analyzes the full frame.
    """

    if frame is None:

        raise ValueError(
            "Frame is empty."
        )


    if frame.size == 0:

        raise ValueError(
            "Frame contains no image data."
        )


    frame_height, frame_width = frame.shape[:2]


    # -----------------------------------------------------
    # BGR -> RGB
    # -----------------------------------------------------

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # -----------------------------------------------------
    # 1. MediaPipe face detection
    # -----------------------------------------------------

    detections = []
    try:
        with detector_lock:
            result = face_detector.detect(
                mp_image
            )
            detections = result.detections or []
    except Exception as mp_err:
        print("MediaPipe detection error:", mp_err)

    boxes = []

    if detections:
        for detection in detections:
            bbox = detection.bounding_box

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
                int(
                    bbox.origin_x +
                    bbox.width
                )
            )

            y2 = min(
                frame_height,
                int(
                    bbox.origin_y +
                    bbox.height
                )
            )

            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2 - x1, y2 - y1))

    # -----------------------------------------------------
    # 2. Haar Cascade fallback if MediaPipe found no faces
    # -----------------------------------------------------

    if not boxes and haar_cascades:
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for cascade in haar_cascades:
                haar_faces = cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.15,
                    minNeighbors=6,
                    minSize=(50, 50)
                )
                if len(haar_faces) > 0:
                    for (hx, hy, hw, hh) in haar_faces:
                        boxes.append((int(hx), int(hy), int(hw), int(hh)))
                    break
        except Exception as haar_err:
            print("Haar cascade fallback error:", haar_err)

    # -----------------------------------------------------
    # Apply Non-Maximum Suppression (Merge duplicate boxes)
    # -----------------------------------------------------

    boxes = apply_nms(boxes)

    predictions = []


    # =====================================================
    # PROCESS EVERY DETECTED FACE
    # =====================================================

    for face_index, (bx, by, bw, bh) in enumerate(
        boxes,
        start=1
    ):

        try:

            if bw < 16 or bh < 16:
                continue

            x1 = max(0, bx)
            y1 = max(0, by)
            x2 = min(frame_width, bx + bw)
            y2 = min(frame_height, by + bh)

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
            # Resize face
            # -------------------------------------------------

            face_input = cv2.resize(
                face_crop,
                (224, 224),
                interpolation=cv2.INTER_AREA
            )


            # -------------------------------------------------
            # Deepfake prediction
            # -------------------------------------------------

            with prediction_lock:

                label, confidence = predict_face(
                    face_input
                )


            # -------------------------------------------------
            # Normalize confidence
            # -------------------------------------------------

            confidence = float(confidence)

            if confidence <= 1:

                confidence_percentage = (
                    confidence * 100
                )

            else:

                confidence_percentage = confidence


            confidence_percentage = max(
                0,
                min(
                    100,
                    confidence_percentage
                )
            )


            # -------------------------------------------------
            # Store prediction
            # -------------------------------------------------

            predictions.append({

                "face": face_index,

                "result": str(label).upper(),

                "confidence": round(
                    confidence_percentage,
                    2
                ),

                "bounding_box": {

                    "x": x1,

                    "y": y1,

                    "width": x2 - x1,

                    "height": y2 - y1,

                    "frame_width": frame_width,

                    "frame_height": frame_height

                }

            })


        except Exception as face_error:

            print(
                f"Face {face_index} processing failed: "
                f"{face_error}"
            )

            continue


    # -----------------------------------------------------
    # 3. Full Frame Fallback if no individual faces found
    # -----------------------------------------------------

    if not predictions and allow_fallback:
        try:
            with prediction_lock:
                label, confidence = predict_face(
                    frame
                )

            confidence_percentage = float(confidence)

            if confidence_percentage <= 1:
                confidence_percentage *= 100

            confidence_percentage = max(
                0,
                min(
                    100,
                    confidence_percentage
                )
            )

            predictions.append({
                "face": 1,
                "result": str(label).upper(),
                "confidence": round(
                    confidence_percentage,
                    2
                ),
                "bounding_box": {
                    "x": 0,
                    "y": 0,
                    "width": frame_width,
                    "height": frame_height,
                    "frame_width": frame_width,
                    "frame_height": frame_height
                }
            })
        except Exception as full_err:
            print("Full image fallback error:", full_err)

    return predictions


# =========================================================
# ROOT
# =========================================================

@app.get("/")
def root():

    return {

        "message": "FakeShield API is running",

        "status": "online",

        "version": "1.0.0"

    }


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health_check():

    return {

        "status": "healthy",

        "service": "FakeShield Backend",

        "face_detector": "ready",

        "deepfake_detector": "ready"

    }


# =========================================================
# IMAGE PREDICTION
# =========================================================

@app.post("/predict")
async def predict_image(
    file: UploadFile = File(...)
):

    try:

        # -------------------------------------------------
        # Read + decode image
        # -------------------------------------------------

        frame = await read_uploaded_image(
            file,
            error_prefix="Image"
        )


        # -------------------------------------------------
        # Analyze (with full-image fallback)
        # -------------------------------------------------

        predictions = analyze_frame(
            frame,
            allow_fallback=True
        )


        # -------------------------------------------------
        # Ensure at least 1 prediction is guaranteed
        # -------------------------------------------------

        if not predictions:
            label, conf = predict_face(frame)
            conf_pct = float(conf) * 100 if float(conf) <= 1 else float(conf)
            conf_pct = max(0, min(100, conf_pct))

            predictions = [{
                "face": 1,
                "result": str(label).upper(),
                "confidence": round(conf_pct, 2),
                "bounding_box": {
                    "x": 0,
                    "y": 0,
                    "width": frame.shape[1],
                    "height": frame.shape[0]
                }
            }]


        # -------------------------------------------------
        # Final response
        # -------------------------------------------------

        return {

            "success": True,

            "faces_detected": len(
                predictions
            ),

            "predictions": predictions,

            "message": (
                f"{len(predictions)} face(s) "
                "analyzed successfully."
            )

        }


    except HTTPException:

        # Keep our intended HTTP errors.

        raise


    except Exception as error:

        print(
            "Image prediction error:",
            error
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "An unexpected error occurred "
                "while analyzing the image."
            )
        )


# =========================================================
# LIVE CAMERA FRAME PREDICTION
# =========================================================

@app.post("/predict-frame")
async def predict_camera_frame(
    file: UploadFile = File(...)
):

    try:

        # -------------------------------------------------
        # Read + decode camera frame
        # -------------------------------------------------

        frame = await read_uploaded_image(
            file,
            error_prefix="Camera frame"
        )


        # -------------------------------------------------
        # Analyze frame (with fallback for partial/cropped faces)
        # -------------------------------------------------

        predictions = analyze_frame(
            frame,
            allow_fallback=True
        )


        # -------------------------------------------------
        # No face
        # -------------------------------------------------

        if not predictions:

            return {

                "success": True,

                "faces_detected": 0,

                "predictions": [],

                "message": "No face detected."

            }


        # -------------------------------------------------
        # Camera response
        # -------------------------------------------------

        return {

            "success": True,

            "faces_detected": len(
                predictions
            ),

            "predictions": predictions,

            "message": (
                f"{len(predictions)} face(s) "
                "analyzed successfully."
            )

        }


    except HTTPException:

        raise


    except Exception as error:

        print(
            "Camera frame prediction error:",
            error
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "An unexpected error occurred "
                "while analyzing the camera frame."
            )
        )