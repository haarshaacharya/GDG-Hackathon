import os
import uvicorn

# Suppress TensorFlow informational logs from spamming STDERR
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["ABSL_LOG_MINIMUM_SEVERITY"] = "2"

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    reload = os.getenv("RELOAD", "false").lower() == "true"
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=reload)

