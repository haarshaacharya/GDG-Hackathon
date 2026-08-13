import { useEffect, useState } from "react";
import "./App.css";

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // Restore image + File after refresh
  useEffect(() => {
    const savedImage = localStorage.getItem("fakeshield_image");
    const savedName = localStorage.getItem("fakeshield_filename");
    const savedType = localStorage.getItem("fakeshield_type");

    if (savedImage) {
      setPreview(savedImage);

      // Convert saved Base64 image back into a File
      fetch(savedImage)
        .then((response) => response.blob())
        .then((blob) => {
          const file = new File(
            [blob],
            savedName || "fakeshield-image.jpg",
            {
              type: savedType || blob.type || "image/jpeg",
            }
          );

          setSelectedFile(file);
        })
        .catch(() => {
          setError("Could not restore the saved image.");
        });
    }
  }, []);

  // Select image
  const handleFileChange = (event) => {
    const file = event.target.files[0];

    if (!file) return;

    if (!file.type.startsWith("image/")) {
      setError("Please select a valid image.");
      return;
    }

    const reader = new FileReader();

    reader.onload = () => {
      const imageData = reader.result;

      setPreview(imageData);

      // Save image information
      localStorage.setItem("fakeshield_image", imageData);
      localStorage.setItem("fakeshield_filename", file.name);
      localStorage.setItem("fakeshield_type", file.type);
    };

    reader.readAsDataURL(file);

    setSelectedFile(file);
    setResult(null);
    setError("");

    // Allow selecting the same image again
    event.target.value = "";
  };

  // Remove image
  const handleRemoveImage = () => {
    setSelectedFile(null);
    setPreview(null);
    setResult(null);
    setError("");

    localStorage.removeItem("fakeshield_image");
    localStorage.removeItem("fakeshield_filename");
    localStorage.removeItem("fakeshield_type");
  };

  // Analyze image
  const handleAnalyze = async () => {
    if (!selectedFile) {
      setError("Please select an image first.");
      return;
    }

    setLoading(true);
    setResult(null);
    setError("");

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Something went wrong.");
      }

      setResult(data);
    } catch (err) {
      setError(
        err.message || "Could not connect to FakeShield backend."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      {/* Navbar */}
      <header className="navbar">
        <div className="logo">FakeShield</div>

        <div className="status">
          <span className="status-dot"></span>
          Backend Online
        </div>
      </header>

      <main className="main-content">
        {/* Hero */}
        <section className="hero">
          <h1>
            AI-Powered
            <span>Deepfake Detection</span>
          </h1>

          <p>
            Upload an image and let FakeShield analyze the detected face using AI.
          </p>
        </section>

        {/* Upload */}
        <section className="upload-card">
          <div className="upload-area">
            {preview ? (
              <div className="preview-container">
                <img
                  src={preview}
                  alt="Selected preview"
                  className="preview-image"
                />

                <button
                  type="button"
                  className="remove-image-button"
                  onClick={handleRemoveImage}
                  aria-label="Remove image"
                  title="Remove image"
                >
                  ×
                </button>
              </div>
            ) : (
              <>
                <div className="upload-icon">+</div>

                <h2>Upload an image</h2>

                <p>Select an image containing a face to analyze.</p>
              </>
            )}

            <input
              type="file"
              accept="image/*"
              id="image-upload"
              onChange={handleFileChange}
              hidden
            />

            {/* Only shown when NO image exists */}
            {!preview && (
              <label htmlFor="image-upload" className="choose-button">
                Choose Image
              </label>
            )}
          </div>

          {/* Analyze button */}
          {selectedFile && (
            <button
              className="analyze-button"
              onClick={handleAnalyze}
              disabled={loading}
            >
              {loading ? "Analyzing..." : "Analyze Image"}
            </button>
          )}

          {/* Error */}
          {error && <div className="error-box">{error}</div>}
        </section>

        {/* Result */}
        {result && (
          <section className="result-card">
            <h2>Analysis Result</h2>

            <div className="result-info">
              <div className="result-item">
                <span>Faces Detected</span>
                <strong>{result.faces_detected}</strong>
              </div>

              {result.predictions &&
                result.predictions.map((prediction, index) => (
                  <div className="prediction" key={index}>
                    <div>
                      <span>Result</span>
                      <strong
                        className={
                          prediction.result === "FAKE" ? "fake" : "real"
                        }
                      >
                        {prediction.result}
                      </strong>
                    </div>

                    <div>
                      <span>Confidence</span>
                      <strong>{prediction.confidence}%</strong>
                    </div>
                  </div>
                ))}
            </div>

            {result.message && (
              <p className="result-message">{result.message}</p>
            )}
          </section>
        )}
      </main>
    </div>
  );
}

export default App;