# 🛡️ FakeShield – Real-Time Deepfake Detection for Live Video Calls
 
**Hack Orbit 2026 | Track 1 — AI & Human Augmentation | PS-02**
 
FakeShield is a real-time deepfake detection system built to protect live video calls from identity fraud and manipulated-face attacks — the kind of threat that offline, post-recording detectors simply can't catch. It combines MediaPipe's multi-face detection with a custom CNN classifier to flag manipulated faces as they appear on screen, with a live confidence overlay so users can see risk in real time.
 
🔗 **Live Demo:** [gdg-hackathon-beta.vercel.app](https://gdg-hackathon-beta.vercel.app/)
 
---
 
## 📌 Problem Statement
 
> **PS-02 | FakeShield: Real-Time Deepfake Detection for Live Video Calls**
> Deepfake attacks during video calls (identity fraud in KYC, financial meetings, academic exams) are a growing threat. Existing detectors operate offline on recorded videos, leaving live calls completely unprotected.
 
**Our goal:** build a real-time deepfake detection system that analyzes incoming video streams frame-by-frame and flags manipulated faces with a visible confidence overlay — fast enough and light enough to run on a standard CPU during an active call.
 
### Target Constraints (from the problem statement)
| Constraint | Target | Status |
| :--- | :--- | :--- |
| Minimum frame rate on standard CPU | ≥ 15 FPS | 🟡 In progress |
| False positive rate on clean video benchmark | ≤ 10% | 🟡 Under validation |
| Video call platform integration | Google Meet / Zoom / Teams (or any accessible platform) | 🟡 Planned |
| Model footprint | Lightweight, real-time capable | ✅ CNN sized for inference speed |
| Output | Live confidence overlay on manipulated faces | ✅ Implemented (image + webcam) |
 
---
 
## 📁 Project Structure
 
```
FakeShield/
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # Main React Application & State Management
│   │   ├── App.css          # Styling & Animations
│   │   ├── main.jsx         # Entry Point
│   │   └── assets/          # Static Assets & Icons
│   │
│   ├── package.json         # React & Vite Dependencies
│   └── vite.config.js       # Vite Configuration
│
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI Application & Endpoints
│   │   ├── services/
│   │   │   └── deepfake_detector.py  # Deepfake Inference Logic
│   │   ├── models/
│   │   │   ├── deepfake_model.keras  # Trained Keras Model
│   │   │   └── face_detector.tflite  # MediaPipe Face Detector Model
│   │   └── __init__.py
│   │
│   ├── requirements.txt     # Python Dependencies
│   └── run.py                # Uvicorn Server Starter
│
├── dataset/                  # Dataset Storage / Reference
│
├── documentation/            # Project Documentation
│   ├── SRS.pdf                # System Requirements Specification
│   ├── Design.pdf             # Design Architecture Document
│   └── PPT.pptx                # Presentation Deck
│
└── README.md                  # Project Overview & Status
```
 
---
 
## 🛠️ Technology Stack
 
| Layer | Technologies Used |
| :--- | :--- |
| **Frontend** | React.js, Vite, HTML5 Canvas, Modern CSS3 |
| **Backend** | Python, FastAPI, Uvicorn, MediaPipe Tasks |
| **AI / Machine Learning** | TensorFlow / Keras (CNN), MediaPipe Face Detection |
| **Computer Vision** | OpenCV (`cv2`), NumPy, PIL |
| **Deployment** | Vercel (frontend) |
 
---
 
## ✨ Core Features
 
- 🖼️ **Image-based Deepfake Analysis** — Upload any image for face extraction and deepfake classification.
- 📹 **Live Webcam Detection** — Real-time camera feed analysis with a live FPS counter.
- 👤 **Multi-Face Detection** — Identifies and bounds multiple faces in a single frame using MediaPipe.
- 🎯 **Confidence Scores** — Per-face real vs. fake probability, plus an overall image risk evaluation.
- 🟦 **Bounding Box Visualization** — Dynamically rendered boxes over each detected face, colored by risk.
- 💾 **Local Storage Persistence** — Preserves session image previews across page refreshes.
- ⚡ **Lightweight Inference Pipeline** — CNN sized for near real-time prediction on CPU-only hardware.
---
 
## 📊 Work Completed Status
 
| Module | Progress | Status Summary |
| :--- | :---: | :--- |
| **Frontend** | 100% | React + Vite UI, image upload/preview, live camera overlay, FPS counter, state management |
| **Backend** | 95% | FastAPI REST endpoints, CORS, MediaPipe integration, image preprocessing, JSON response |
| **AI Model** | 100% | Pre-trained CNN model loading, input normalization (224×224), prediction pipeline |
| **Face Detection** | 100% | MediaPipe Face Detector, bounding box scaling, crop & invalid-face handling |
| **Camera Module** | 95% | Live stream loop, async analysis, abort controller, session safety |
| **API Integration** | 100% | Connected `/predict`, `/predict-frame`, `/health`, and root endpoints |
| **Live Call Integration** | 0% | Browser extension / virtual camera driver — not yet started |
| **Testing** | 80% | Core logic & pipeline tested; FPS benchmarking and false-positive validation ongoing |
| **Documentation** | 65% | Structure & status finalized; detailed API docs in progress |
 
**Overall Project Completion: ≈ 90%** (core detection pipeline is production-ready; live video-call integration is the primary remaining gap against PS-02)
 
---
 
## 🔌 API Endpoints Summary
 
| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Root status and welcome message |
| `GET` | `/health` | API health check & model loading status |
| `POST` | `/predict` | Image file upload analysis (`multipart/form-data`) |
| `POST` | `/predict-frame` | Real-time camera frame string/blob prediction |
 
---
 
## 🚀 Getting Started
 
### Backend Setup
```bash
cd backend
python -m venv venv
 
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
 
pip install -r requirements.txt
python run.py
# Backend runs at http://localhost:8000
```
 
### Frontend Setup
```bash
cd frontend
npm install
npm run dev
# Frontend runs at http://localhost:5173
```
 
---
 
## 🎯 Remaining Roadmap (mapped to PS-02 constraints)
 
1. **Live Call Integration** — Build a browser extension or virtual camera driver so FakeShield can intercept and analyze frames from Google Meet, Zoom, or Microsoft Teams in real time.
2. **Performance Benchmarking** — Validate ≥15 FPS throughput on standard CPU hardware; optimize the inference pipeline (model quantization / frame skipping) if needed.
3. **False Positive Validation** — Run the clean-video benchmark and confirm the false positive rate stays at or below 10%.
4. **Security Hardening** — Add rate limiting, API token verification, and payload size restriction middleware.
5. **Deployment** — Containerize with `docker-compose.yml` for single-command production deployment.
6. **Final Deliverables** — Demo video, user manual, and presentation slide deck for submission.
---
 
## 👥 Team
 
Haarsh Aacharya 
Jugal Shah
Pushparajsinh Gohil
Parthrajsinh Gohil
 
---
 
## 📄 Submission Checklist (Hack Orbit 2026)
 
- [x] Public GitHub repository with descriptive README
- [ ] 3-minute demo video (YouTube/Drive)
- [ ] 1-page project abstract (Google Form)
- [x] Working, deployed prototype — [live link](https://gdg-hackathon-beta.vercel.app/)
 
