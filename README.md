# 🛡️ FakeShield – AI-Powered Deepfake Detection System

FakeShield is a state-of-the-art deepfake detection application built for real-time analysis of images and live webcam streams. Utilizing MediaPipe for robust multi-face detection and custom CNN deep learning models for classification, FakeShield delivers instant confidence scores and visual bounding boxes.

---

## 📁 Project Structure
## project

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
│   │   │   └── deepfake_detector.py # Deepfake Inference Logic
│   │   ├── models/
│   │   │   ├── deepfake_model.keras # Trained Keras Model
│   │   │   └── face_detector.tflite # MediaPipe Face Detector Model
│   │   └── __init__.py
│   │
│   ├── requirements.txt     # Python Dependencies
│   └── run.py               # Uvicorn Server Starter
│
├── dataset/                 # Dataset Storage / Reference
│
├── documentation/           # Project Documentation
│   ├── SRS.pdf              # System Requirements Specification
│   ├── Design.pdf           # Design Architecture Document
│   └── PPT.pptx             # Presentation Deck
│
└── README.md                # Project Overview & Status
```

---

## 🛠️ Technology Stack

| Layer | Technologies Used |
| :--- | :--- |
| **Frontend** | React.js, Vite, HTML5 Canvas, Modern CSS3 |
| **Backend** | Python, FastAPI, Uvicorn, MediaPipe Tasks |
| **AI / Machine Learning** | TensorFlow / Keras (CNN), MediaPipe Face Detection |
| **Computer Vision** | OpenCV (`cv2`), NumPy, PIL |

---

## ✨ Core Features

- 🖼️ **Image-based Deepfake Analysis**: Upload any image for face extraction and deepfake classification.
- 📹 **Live Webcam Detection**: Real-time camera feed analysis with live FPS counter.
- 👤 **Multi-Face Detection**: Identifies and bounds multiple faces in a single frame.
- 🎯 **Confidence Scores**: Provides per-face real vs. fake probability scores and overall image risk evaluation.
- 🟦 **Bounding Box Visualization**: Renders bounding boxes dynamically over detected faces.
- 💾 **Local Storage Persistence**: Preserves session image previews across page refreshes.

---

## 📊 Work Completed Status

| Module | Progress | Status Summary |
| :--- | :---: | :--- |
| **Frontend** | **100%** | React + Vite UI, Image upload/preview, Live camera overlay, FPS counter, State management |
| **Backend** | **95%** | FastAPI REST endpoints, CORS, MediaPipe integration, Image preprocessing, JSON response |
| **AI Model** | **100%** | Pre-trained CNN model loading, input normalization (224x224), prediction pipeline |
| **Face Detection** | **100%** | MediaPipe Face Detector, bounding box scaling, crop & invalid face handling |
| **Camera Module** | **95%** | Live stream loop, async analysis, abort controller, session safety |
| **API Integration** | **100%** | Connected `/predict`, `/predict-frame`, `/health`, and root endpoints |
| **Testing** | **80%** | Core logic & pipeline tested; cross-device & stress testing ongoing |
| **Documentation** | **50%** | Structure & status finalized; detailed API docs in progress |

**Overall Project Completion: ≈ 95%**

---

## 🔌 API Endpoints Summary

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Root status and welcoming message |
| `GET` | `/health` | API health check & model loading status |
| `POST` | `/predict` | Image file upload analysis (`multipart/form-data`) |
| `POST` | `/predict-frame` | Real-time camera frame string/blob prediction |

---

## 🚀 Getting Started
## start

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

## 🎯 Remaining Roadmap

1. **Testing**: Expand lighting condition tests, edge cases with extreme facial angles, and low latency optimization.
2. **Security**: Add rate limiting, API token verification, and payload size restriction middleware.
3. **Deployment**: Containerize with `docker-compose.yml` for single-command production deployment (Vercel/Render).
4. **Final Deliverables**: Complete demo video, user manual, and presentation slide deck.
