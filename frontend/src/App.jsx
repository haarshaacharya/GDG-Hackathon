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

      localStorage.setItem(
        "fakeshield_image",
        imageData
      );

      localStorage.setItem(
        "fakeshield_filename",
        file.name
      );

      localStorage.setItem(
        "fakeshield_type",
        file.type
      );
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

    formData.append(
      "file",
      selectedFile
    );

    try {
      const response = await fetch(
        "http://127.0.0.1:8000/predict",
        {
          method: "POST",
          body: formData,
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || "Something went wrong."
        );
      }

      setResult(data);
    } catch (err) {
      setError(
        err.message ||
          "Could not connect to FakeShield backend."
      );
    } finally {
      setLoading(false);
    }
  };

  // Get overall result
  const getOverallResult = () => {
    if (
      !result ||
      !result.predictions ||
      result.predictions.length === 0
    ) {
      return null;
    }

    const fakeFaces = result.predictions.filter(
      (prediction) =>
        prediction.result === "FAKE"
    ).length;

    if (fakeFaces > 0) {
      return "FAKE";
    }

    return "REAL";
  };

  // Get overall confidence
  const getAverageConfidence = () => {
    if (
      !result ||
      !result.predictions ||
      result.predictions.length === 0
    ) {
      return 0;
    }

    const total = result.predictions.reduce(
      (sum, prediction) =>
        sum + Number(prediction.confidence || 0),
      0
    );

    return Math.round(
      total / result.predictions.length
    );
  };

  const overallResult = getOverallResult();
  const averageConfidence = getAverageConfidence();

  return (
    <div className="app">

      {/* =================================================
          NAVBAR
          ================================================= */}

      <header className="navbar">

        <div className="logo">
          FakeShield
        </div>

        <div className="status">
          <span className="status-dot"></span>
          Backend Online
        </div>

      </header>


      <main className="main-content">

        {/* =================================================
            HERO
            ================================================= */}

        <section className="hero">

          <h1>
            AI-Powered
            <span>Deepfake Detection</span>
          </h1>

          <p>
            Upload an image and let FakeShield
            analyze the detected face using AI.
          </p>

        </section>


        {/* =================================================
            UPLOAD CARD
            ================================================= */}

        <section className="upload-card">

          <div className="upload-area">

            {preview ? (

              <div className="preview-container">

                <img
                  src={preview}
                  alt="Selected preview"
                  className="preview-image"
                />

                {/* Remove button */}
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

                <div className="upload-icon">
                  +
                </div>

                <h2>
                  Upload an image
                </h2>

                <p>
                  Select an image containing a face
                  to analyze.
                </p>

              </>

            )}


            <input
              type="file"
              accept="image/*"
              id="image-upload"
              onChange={handleFileChange}
              hidden
            />


            {!preview && (

              <label
                htmlFor="image-upload"
                className="choose-button"
              >
                Choose Image
              </label>

            )}

          </div>


          {/* =================================================
              ANALYZE BUTTON
              ================================================= */}

          {selectedFile && (

            <button
              className="analyze-button"
              onClick={handleAnalyze}
              disabled={loading}
            >

              {loading ? (

                <span className="analyzing-content">

                  <span className="loading-spinner"></span>

                  <span>
                    Analyzing Image...
                  </span>

                </span>

              ) : (

                "Analyze Image"

              )}

            </button>

          )}


          {/* =================================================
              ERROR
              ================================================= */}

          {error && (

            <div className="error-box">

              <div className="error-icon">
                !
              </div>

              <div>
                <strong>
                  Analysis Error
                </strong>

                <p>
                  {error}
                </p>
              </div>

            </div>

          )}

        </section>


        {/* =================================================
            ANALYZING STATE
            ================================================= */}

        {loading && (

          <section className="scanning-card">

            <div className="scanner-icon">

              <span></span>

            </div>

            <div className="scanning-text">

              <h2>
                AI is analyzing the image
              </h2>

              <p>
                Detecting faces and checking
                for manipulation...
              </p>

            </div>

            <div className="scan-progress">

              <div className="scan-progress-bar"></div>

            </div>

          </section>

        )}


        {/* =================================================
            RESULT
            ================================================= */}

        {result && !loading && (

          <section className="result-card">

            {/* Result header */}

            <div className="result-header">

              <div>

                <span className="result-label">
                  ANALYSIS COMPLETE
                </span>

                <h2>
                  Analysis Result
                </h2>

              </div>

              <div
                className={`overall-badge ${
                  overallResult === "FAKE"
                    ? "badge-fake"
                    : overallResult === "REAL"
                    ? "badge-real"
                    : "badge-neutral"
                }`}
              >
                {overallResult || "NO FACE"}
              </div>

            </div>


            {/* =================================================
                SUMMARY
                ================================================= */}

            {result.faces_detected > 0 ? (

              <>

                <div className="result-summary">

                  <div className="summary-card">

                    <span>
                      Faces Detected
                    </span>

                    <strong>
                      {result.faces_detected}
                    </strong>

                  </div>


                  <div className="summary-card">

                    <span>
                      Average Confidence
                    </span>

                    <strong>
                      {averageConfidence}%
                    </strong>

                  </div>


                  <div className="summary-card">

                    <span>
                      Faces Flagged
                    </span>

                    <strong className="flagged-number">
                      {
                        result.predictions
                          ? result.predictions.filter(
                              (prediction) =>
                                prediction.result === "FAKE"
                            ).length
                          : 0
                      }
                    </strong>

                  </div>

                </div>


                {/* =================================================
                    FACE RESULTS
                    ================================================= */}

                <div className="face-results">

                  <div className="face-results-title">
                    Face-by-Face Analysis
                  </div>

                  {result.predictions &&
                    result.predictions.map(
                      (prediction, index) => {

                        const isFake =
                          prediction.result === "FAKE";

                        return (

                          <div
                            className="face-result"
                            key={index}
                          >

                            {/* Face number */}

                            <div className="face-number">
                              {index + 1}
                            </div>


                            {/* Result */}

                            <div className="face-result-main">

                              <span className="face-result-label">
                                Face {index + 1}
                              </span>

                              <strong
                                className={
                                  isFake
                                    ? "fake"
                                    : "real"
                                }
                              >
                                {prediction.result}
                              </strong>

                            </div>


                            {/* Confidence */}

                            <div className="face-confidence">

                              <div className="confidence-top">

                                <span>
                                  Confidence
                                </span>

                                <strong>
                                  {prediction.confidence}%
                                </strong>

                              </div>

                              <div className="confidence-bar">

                                <div
                                  className={
                                    `confidence-fill ${
                                      isFake
                                        ? "confidence-fake"
                                        : "confidence-real"
                                    }`
                                  }
                                  style={{
                                    width: `${Math.min(
                                      Number(
                                        prediction.confidence
                                      ) || 0,
                                      100
                                    )}%`,
                                  }}
                                ></div>

                              </div>

                            </div>

                          </div>

                        );
                      }
                    )}

                </div>

              </>

            ) : (

              /* =================================================
                 NO FACE
                 ================================================= */

              <div className="no-face-state">

                <div className="no-face-icon">
                  ?
                </div>

                <h3>
                  No Face Detected
                </h3>

                <p>
                  FakeShield couldn't find a detectable
                  face in this image. Try uploading a
                  clearer image where the face is visible.
                </p>

              </div>

            )}


            {/* Backend message */}

            {result.message && result.faces_detected > 0 && (

              <p className="result-message">
                {result.message}
              </p>

            )}

          </section>

        )}

      </main>

    </div>
  );
}

export default App;