from fastapi import FastAPI

app = FastAPI(
    title="FakeShield AI",
    description="Real-Time Deepfake Detection API",
    version="1.0.0"
)


@app.get("/")
def root():
    return {
        "status": "online",
        "message": "FakeShield AI Backend is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }