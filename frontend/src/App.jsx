import { useEffect, useRef, useState } from "react";
import "./App.css";

const API_BASE_URL =
  import.meta.env.VITE_API_URL ||
  "https://gdg-hackathon-thwr.onrender.com";


function App() {
  // =====================================================
  // IMAGE STATES
  // =====================================================

  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // =====================================================
  // CAMERA STATES
  // =====================================================

  const [cameraActive, setCameraActive] = useState(false);
  const [showCameraOverlay, setShowCameraOverlay] = useState(false);

  const [cameraPredictions, setCameraPredictions] = useState([]);
  const [cameraFaces, setCameraFaces] = useState(0);
  const [cameraError, setCameraError] = useState("");

  const [fps, setFps] = useState(0);
  const [cameraAnalyzing, setCameraAnalyzing] = useState(false);

  // =====================================================
  // CAMERA REFS
  // =====================================================

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);

  const cameraTimerRef = useRef(null);
  const inFlightRef = useRef(0);

  // Abort current AI request when camera stops
  const cameraAbortRef = useRef(null);

  // Used to invalidate old camera loops
  const cameraSessionRef = useRef(0);

  // Actual AI analysis FPS tracking
  const analysisTimesRef = useRef([]);

  // Target AI analysis rate
  const AI_INTERVAL = 15;

  // =====================================================
  // RESTORE IMAGE AFTER REFRESH
  // =====================================================

  useEffect(() => {
    const savedImage =
      localStorage.getItem("fakeshield_image");

    const savedName =
      localStorage.getItem("fakeshield_filename");

    const savedType =
      localStorage.getItem("fakeshield_type");

    if (savedImage) {
      setPreview(savedImage);

      fetch(savedImage)
        .then((response) => response.blob())
        .then((blob) => {
          const file = new File(
            [blob],
            savedName || "fakeshield-image.jpg",
            {
              type:
                savedType ||
                blob.type ||
                "image/jpeg",
            }
          );

          setSelectedFile(file);
        })
        .catch(() => {
          setError(
            "Could not restore the saved image."
          );
        });
    }
  }, []);

  // =====================================================
  // CLEAN EVERYTHING WHEN PAGE CLOSES
  // =====================================================

  useEffect(() => {
    return () => {
      stopCameraInternal();
    };
  }, []);

  // =====================================================
  // SELECT IMAGE
  // =====================================================

  const handleFileChange = (event) => {
    const file = event.target.files[0];

    if (!file) return;

    if (!file.type.startsWith("image/")) {
      setError(
        "Please select a valid image."
      );

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

    event.target.value = "";
  };

  // =====================================================
  // REMOVE IMAGE
  // =====================================================

  const handleRemoveImage = () => {
    setSelectedFile(null);
    setPreview(null);
    setResult(null);
    setError("");

    localStorage.removeItem(
      "fakeshield_image"
    );

    localStorage.removeItem(
      "fakeshield_filename"
    );

    localStorage.removeItem(
      "fakeshield_type"
    );
  };

  // =====================================================
  // ANALYZE IMAGE
  // =====================================================

  const handleAnalyze = async () => {
    if (!selectedFile) {
      setError(
        "Please select an image first."
      );

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
        `${API_BASE_URL}/predict`,
        {
          method: "POST",
          body: formData,
        }
      );

      const data =
        await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail ||
          "Something went wrong."
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

  // =====================================================
  // RESULT HELPERS
  // =====================================================

  const getImageStats = () => {
    if (
      !result ||
      !result.predictions ||
      result.predictions.length === 0
    ) {
      return {
        overallResult: "UNKNOWN",
        averageConfidence: 0,
        flaggedFaces: 0,
        category: "Analysis Complete",
        signals: [],
      };
    }

    const predictions = result.predictions;

    const totalConfidence = predictions.reduce(
      (total, prediction) =>
        total + Number(prediction.confidence || 0),
      0
    );

    const averageConfidence =
      totalConfidence / predictions.length;

    const flaggedFaces = predictions.filter(
      (prediction) =>
        String(prediction.result).toUpperCase() === "FAKE"
    ).length;

    const realFaces = predictions.filter(
      (prediction) =>
        String(prediction.result).toUpperCase() === "REAL"
    ).length;

    let overallResult = "UNKNOWN";

    if (flaggedFaces > 0) {
      overallResult = "FAKE";
    } else if (realFaces > 0) {
      overallResult = "REAL";
    }

    const category =
      result.category ||
      (overallResult === "FAKE"
        ? "AI-Generated / Deepfake Image"
        : "Real Human / Mobile Camera Photo");

    const signals = result.signals || predictions[0]?.signals || [];

    return {
      overallResult,
      averageConfidence,
      flaggedFaces,
      category,
      signals,
    };
  };

  const imageStats = getImageStats();

  // =====================================================
  // CLEAR CAMERA TIMER
  // =====================================================

  const clearCameraTimer = () => {
    if (cameraTimerRef.current) {
      clearTimeout(
        cameraTimerRef.current
      );

      cameraTimerRef.current = null;
    }
  };

  // =====================================================
  // STOP CAMERA INTERNAL
  // =====================================================

  const stopCameraInternal = () => {
    // Invalidate previous camera loop
    cameraSessionRef.current += 1;

    clearCameraTimer();

    // Abort current backend request
    if (cameraAbortRef.current) {
      cameraAbortRef.current.abort();
      cameraAbortRef.current = null;
    }

    // Stop camera tracks
    if (streamRef.current) {
      streamRef.current
        .getTracks()
        .forEach((track) => {
          try {
            track.stop();
          } catch {
            // Ignore already stopped tracks
          }
        });

      streamRef.current = null;
    }

    // Reset video
    if (videoRef.current) {
      try {
        videoRef.current.pause();
      } catch {
        // Ignore
      }

      videoRef.current.srcObject = null;
    }

    inFlightRef.current = 0;

    analysisTimesRef.current = [];

    document.body.style.overflow = "";
  };

  // =====================================================
  // START CAMERA
  // =====================================================

  const startCamera = async () => {
    // Prevent duplicate camera sessions
    if (streamRef.current) {
      return;
    }

    try {
      setCameraError("");
      setCameraPredictions([]);
      setCameraFaces(0);
      setFps(0);
      setCameraAnalyzing(false);

      // New session ID
      cameraSessionRef.current += 1;

      const currentSession =
        cameraSessionRef.current;

      analysisTimesRef.current = [];
      inFlightRef.current = 0;

      if (
        !navigator.mediaDevices ||
        !navigator.mediaDevices.getUserMedia
      ) {
        throw new Error(
          "Camera is not supported by this browser."
        );
      }

      const stream =
        await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: "user",

            width: {
              ideal: 640,
            },

            height: {
              ideal: 480,
            },

            frameRate: {
              ideal: 30,
              min: 24,
            },
          },

          audio: false,
        });

      // Check whether user stopped camera
      // while permission dialog was open.
      if (
        currentSession !==
        cameraSessionRef.current
      ) {
        stream
          .getTracks()
          .forEach((track) =>
            track.stop()
          );

        return;
      }

      streamRef.current = stream;

      setCameraActive(true);
      setShowCameraOverlay(true);

      // Camera overlay should not move page
      document.body.style.overflow = "hidden";

      // Attach stream after render
      requestAnimationFrame(
        async () => {
          const video =
            videoRef.current;

          if (!video) return;

          video.srcObject = stream;

          try {
            await video.play();
          } catch (playError) {
            console.error(
              "Video play error:",
              playError
            );
          }

          if (
            streamRef.current &&
            currentSession ===
              cameraSessionRef.current
          ) {
            startFrameLoop(
              currentSession
            );
          }
        }
      );
    } catch (err) {
      console.error(
        "Camera start error:",
        err
      );

      stopCameraInternal();

      setCameraError(
        err.message ||
        "Could not access your camera."
      );

      setCameraActive(false);
      setShowCameraOverlay(false);
      setCameraAnalyzing(false);
    }
  };

  // =====================================================
  // STOP CAMERA
  // =====================================================

  const stopCamera = () => {
    stopCameraInternal();

    setCameraActive(false);
    setShowCameraOverlay(false);

    setCameraAnalyzing(false);
    setCameraPredictions([]);
    setCameraFaces(0);
    setFps(0);
  };

  // =====================================================
  // UPDATE REAL AI FPS
  // =====================================================

  const updateAnalysisFps = () => {
    const now = performance.now();

    analysisTimesRef.current.push(now);

    // Keep only last 2 seconds
    analysisTimesRef.current =
      analysisTimesRef.current.filter(
        (time) =>
          now - time <= 2000
      );

    const times =
      analysisTimesRef.current;

    if (times.length <= 1) {
      setFps(0);
      return;
    }

    const firstTime = times[0];
    const lastTime =
      times[times.length - 1];

    const elapsed =
      lastTime - firstTime;

    if (elapsed <= 0) {
      return;
    }

    const actualFps =
      (times.length - 1) /
      (elapsed / 1000);

    setFps(
      Math.min(
        30,
        Math.round(
          actualFps
        )
      )
    );
  };

  // =====================================================
  // CAMERA FRAME LOOP
  // =====================================================

  const startFrameLoop = (
    sessionId
  ) => {
    const processFrame = async () => {
      // -------------------------------------------------
      // SESSION VALIDATION
      // -------------------------------------------------

      if (
        sessionId !==
        cameraSessionRef.current
      ) {
        return;
      }

      if (!streamRef.current) {
        return;
      }

      // -------------------------------------------------
      // PIPELINE IN-FLIGHT LIMIT (Allows up to 3 concurrent requests)
      // -------------------------------------------------

      if (inFlightRef.current >= 3) {
        cameraTimerRef.current =
          setTimeout(
            processFrame,
            10
          );

        return;
      }

      const video =
        videoRef.current;

      const canvas =
        canvasRef.current;

      if (
        !video ||
        !canvas ||
        video.readyState < 2 ||
        video.videoWidth === 0 ||
        video.videoHeight === 0
      ) {
        cameraTimerRef.current =
          setTimeout(
            processFrame,
            50
          );

        return;
      }

      inFlightRef.current += 1;

      setCameraAnalyzing(true);

      // Schedule next frame loop for 14-15 FPS
      cameraTimerRef.current =
        setTimeout(
          processFrame,
          20
        );

      let requestSucceeded = false;

      try {
        // ------------------------------------------------
        // VIDEO DIMENSIONS
        // ------------------------------------------------

        const width =
          video.videoWidth;

        const height =
          video.videoHeight;

        // ------------------------------------------------
        // LIMIT BACKEND IMAGE SIZE FOR MAXIMUM THROUGHPUT
        // ------------------------------------------------

        const maxWidth = 360;

        let captureWidth = width;
        let captureHeight = height;

        if (width > maxWidth) {
          captureWidth = maxWidth;

          captureHeight =
            Math.round(
              height *
              (maxWidth / width)
            );
        }

        canvas.width =
          captureWidth;

        canvas.height =
          captureHeight;

        const context =
          canvas.getContext("2d", {
            willReadFrequently: false,
          });

        if (!context) {
          throw new Error(
            "Could not access camera canvas."
          );
        }

        // ------------------------------------------------
        // CAPTURE FRAME
        // ------------------------------------------------

        context.drawImage(
          video,
          0,
          0,
          captureWidth,
          captureHeight
        );

        // ------------------------------------------------
        // JPEG (High compression for instant transmission)
        // ------------------------------------------------

        const blob =
          await new Promise(
            (resolve) =>
              canvas.toBlob(
                resolve,
                "image/jpeg",
                0.40
              )
          );

        if (!blob) {
          throw new Error(
            "Could not capture camera frame."
          );
        }

        // ------------------------------------------------
        // CREATE FORM DATA
        // ------------------------------------------------

        const formData =
          new FormData();

        formData.append(
          "file",
          blob,
          "camera-frame.jpg"
        );

        // ------------------------------------------------
        // ABORT CONTROLLER
        // ------------------------------------------------

        const controller =
          new AbortController();

        cameraAbortRef.current =
          controller;

        // ------------------------------------------------
        // BACKEND REQUEST
        // ------------------------------------------------

        const response =
          await fetch(
            `${API_BASE_URL}/predict-frame`,
            {
              method: "POST",
              body: formData,
              signal:
                controller.signal,
            }
          );

        const data =
          await response.json();

        // ------------------------------------------------
        // SESSION VALIDATION
        // ------------------------------------------------

        if (
          sessionId !==
          cameraSessionRef.current
        ) {
          return;
        }

        if (!response.ok) {
          throw new Error(
            data.detail ||
            "Camera analysis failed."
          );
        }

        // ------------------------------------------------
        // UPDATE RESULTS
        // ------------------------------------------------

        const predictions =
          data.predictions || [];

        setCameraPredictions(
          predictions
        );

        setCameraFaces(
          Number(
            data.faces_detected || 0
          )
        );

        // ------------------------------------------------
        // UPDATE ACTUAL AI FPS
        // ------------------------------------------------

        updateAnalysisFps();

        requestSucceeded = true;
      } catch (err) {
        // Abort is expected when stopping camera
        if (
          err.name ===
          "AbortError"
        ) {
          return;
        }

        console.error(
          "Camera analysis error:",
          err
        );

        if (
          sessionId ===
          cameraSessionRef.current
        ) {
          setCameraError(
            err.message ||
            "Camera analysis failed."
          );
        }
      } finally {
        if (
          cameraAbortRef.current
        ) {
          cameraAbortRef.current =
            null;
        }

        inFlightRef.current =
          Math.max(
            0,
            inFlightRef.current - 1
          );

        if (
          sessionId ===
            cameraSessionRef.current &&
          streamRef.current
        ) {
          setCameraAnalyzing(
            inFlightRef.current > 0
          );
        }
      }
    };

    processFrame();
  };

  // =====================================================
  // RENDER
  // =====================================================

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

            <span>
              Deepfake Detection
            </span>

          </h1>

          <p>
            Upload an image or use your camera
            to detect deepfakes using AI.
          </p>

        </section>


        {/* =================================================
            IMAGE UPLOAD
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

                <button
                  type="button"
                  className="remove-image-button"
                  onClick={
                    handleRemoveImage
                  }
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
                  Select an image containing
                  a face to analyze.
                </p>

              </>

            )}


            <input
              type="file"
              accept="image/*"
              id="image-upload"
              onChange={
                handleFileChange
              }
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


          {selectedFile && (

            <button
              className="analyze-button"
              onClick={handleAnalyze}
              disabled={loading}
            >

              {loading
                ? "Analyzing..."
                : "Analyze Image"}

            </button>

          )}


          {error && (

            <div className="error-box">
              {error}
            </div>

          )}

        </section>


        {/* =================================================
            IMAGE RESULT
            ================================================= */}

        {result && (

          <section className="result-card">

            {/* RESULT HEADER */}

            <div className="analysis-header">

              <div>

                <div className="analysis-eyebrow-row">
                  <span className="analysis-eyebrow">
                    ANALYSIS COMPLETE
                  </span>
                  {imageStats.category && (
                    <span className={`category-badge ${imageStats.overallResult === "FAKE" ? "badge-fake" : "badge-real"}`}>
                      {imageStats.overallResult === "FAKE" ? "🤖 " : "📸 "}
                      {imageStats.category}
                    </span>
                  )}
                </div>

                <h2>
                  Analysis Result
                </h2>

              </div>


              <div
                className={`overall-result ${
                  imageStats.overallResult ===
                  "FAKE"
                    ? "fake"
                    : imageStats.overallResult ===
                      "REAL"
                      ? "real"
                      : ""
                }`}
              >
                {imageStats.overallResult}
              </div>

            </div>


            {/* SUMMARY */}

            <div className="analysis-summary">

              <div className="summary-card">

                <span>
                  FACES DETECTED
                </span>

                <strong>
                  {result.faces_detected ||
                    0}
                </strong>

              </div>


              <div className="summary-card">

                <span>
                  AVERAGE CONFIDENCE
                </span>

                <strong>
                  {imageStats.averageConfidence.toFixed(
                    2
                  )}
                  %
                </strong>

              </div>


              <div className="summary-card">

                <span>
                  FACES FLAGGED
                </span>

                <strong
                  className={
                    imageStats.flaggedFaces >
                    0
                      ? "fake"
                      : "real"
                  }
                >
                  {imageStats.flaggedFaces}
                </strong>

              </div>

            </div>


            {/* FORENSIC SIGNALS & EVIDENCE */}

            {imageStats.signals && imageStats.signals.length > 0 && (

              <div className="forensic-signals-section">

                <div className="signals-header">
                  <span className="signals-icon">🔬</span>
                  <h4>Forensic Verification & Detection Signals</h4>
                </div>

                <div className="signals-grid">
                  {imageStats.signals.map((signal, sIdx) => {
                    const isAiSignal = signal.toLowerCase().includes("ai") ||
                                       signal.toLowerCase().includes("synthetic") ||
                                       signal.toLowerCase().includes("artifact") ||
                                       signal.toLowerCase().includes("irregularit");
                    return (
                      <div
                        key={`sig-${sIdx}`}
                        className={`signal-item ${isAiSignal ? "signal-ai" : "signal-real"}`}
                      >
                        <span className="signal-bullet">
                          {isAiSignal ? "⚠" : "✓"}
                        </span>
                        <span className="signal-text">{signal}</span>
                      </div>
                    );
                  })}
                </div>

              </div>

            )}


            {/* FACE BY FACE */}

            <div className="face-analysis">

              <div className="face-analysis-title">

                <div>

                  <span>
                    FACE-BY-FACE ANALYSIS
                  </span>

                  <h3>
                    Detected Faces
                  </h3>

                </div>

                <span className="face-count">
                  {result.predictions
                    ?.length || 0}{" "}
                  Faces
                </span>

              </div>


              <div className="face-list">

                {result.predictions &&
                  result.predictions.map(
                    (
                      prediction,
                      index
                    ) => {

                      const isFake =
                        String(
                          prediction.result
                        ).toUpperCase() ===
                        "FAKE";

                      return (

                        <div
                          className="face-result-card"
                          key={`${prediction.face}-${index}`}
                        >

                          <div className="face-number">
                            {prediction.face ||
                              index + 1}
                          </div>


                          <div className="face-result-main">

                            <div className="face-info-block">
                              <span className="face-label">
                                Face{" "}
                                {prediction.face ||
                                  index + 1}
                              </span>

                              {prediction.category && (
                                <span className="face-sub-category">
                                  {prediction.category}
                                </span>
                              )}
                            </div>


                            <strong
                              className={
                                isFake
                                  ? "fake"
                                  : "real"
                              }
                            >
                              {
                                prediction.result
                              }
                            </strong>

                          </div>


                          <div className="face-confidence">

                            <span>
                              CONFIDENCE
                            </span>

                            <strong>
                              {
                                prediction.confidence
                              }
                              %
                            </strong>

                          </div>

                        </div>

                      );

                    }
                  )}

              </div>

            </div>


            {/* MESSAGE */}

            {result.message && (

              <p className="result-message">
                {result.message}
              </p>

            )}

          </section>

        )}

      </main>


      {/* =================================================
          FLOATING CAMERA BUTTON
          ================================================= */}

      {!showCameraOverlay && (

        <button
          type="button"
          className="floating-camera-btn"
          onClick={startCamera}
        >

          <span className="camera-button-icon">
            ◉
          </span>

          <span>
            Live Camera
          </span>

        </button>

      )}


      {/* =================================================
          CAMERA OVERLAY
          ================================================= */}

      {showCameraOverlay && (

        <div className="camera-overlay">

          <div className="camera-overlay-card">

            {/* CAMERA HEADER */}

            <div className="camera-overlay-header">

              <div>

                <div className="camera-title-row">

                  <span className="camera-live-dot"></span>

                  <span className="camera-live-text">
                    LIVE DETECTION
                  </span>

                </div>


                <h2>
                  Live Camera Detection
                </h2>


                <p>
                  Real-time AI deepfake detection
                </p>

              </div>


              <button
                type="button"
                className="camera-close-btn"
                onClick={stopCamera}
                aria-label="Close camera"
                title="Close camera"
              >
                ×
              </button>

            </div>


            {/* CAMERA VIDEO */}

            <div className="camera-video-wrapper">

              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="camera-video"
              />


              {/* CAMERA SCAN EFFECT */}

              {cameraAnalyzing && (
                <div className="camera-scan-line"></div>
              )}


              {/* CORNERS */}

              <div className="camera-corner camera-corner-tl"></div>

              <div className="camera-corner camera-corner-tr"></div>

              <div className="camera-corner camera-corner-bl"></div>

              <div className="camera-corner camera-corner-br"></div>


              {/* FACE BOXES */}

              {cameraActive &&
                cameraPredictions.map(
                  (
                    prediction,
                    index
                  ) => {

                    const box =
                      prediction.bounding_box;

                    if (!box) {
                      return null;
                    }

                    const video =
                      videoRef.current;

                    if (!video) {
                      return null;
                    }

                    // Use the frame dimensions sent by backend to ensure exact positioning
                    const boxFrameW =
                      box.frame_width ||
                      video.videoWidth ||
                      640;

                    const boxFrameH =
                      box.frame_height ||
                      video.videoHeight ||
                      480;

                    const left =
                      (box.x /
                        boxFrameW) *
                      100;

                    const top =
                      (box.y /
                        boxFrameH) *
                      100;

                    const width =
                      (box.width /
                        boxFrameW) *
                      100;

                    const height =
                      (box.height /
                        boxFrameH) *
                      100;

                    const isFake =
                      String(
                        prediction.result
                      ).toUpperCase() ===
                      "FAKE";

                    /*
                     * Camera is displayed as selfie mirror,
                     * so horizontal coordinate is mirrored.
                     */
                    const mirroredLeft =
                      100 -
                      left -
                      width;

                    return (

                      <div
                        key={`${prediction.face}-${index}`}
                        className={`camera-face-box ${
                          isFake
                            ? "fake"
                            : "real"
                        }`}
                        style={{
                          left: `${mirroredLeft}%`,
                          top: `${top}%`,
                          width: `${width}%`,
                          height: `${height}%`,
                        }}
                      >

                        <div className="camera-face-label">

                          Face{" "}
                          {prediction.face}

                          {" · "}

                          {
                            prediction.result
                          }

                          {" · "}

                          {
                            prediction.confidence
                          }
                          %

                        </div>

                      </div>

                    );
                  }
                )}


              {/* NO FACE MESSAGE */}

              {cameraActive &&
                cameraFaces === 0 && (

                  <div className="camera-no-face">
                    No face detected
                  </div>

                )}

            </div>


            {/* CAMERA STATS */}

            <div className="camera-overlay-footer">

              <div className="camera-stat">

                <span>
                  FACES DETECTED
                </span>

                <strong>
                  {cameraFaces}
                </strong>

              </div>


              <div className="camera-stat">

                <span>
                  AI PROCESSING
                </span>

                <strong>
                  {fps > 0
                    ? `${fps} FPS`
                    : "WAITING"}
                </strong>

              </div>


              <div className="camera-stat">

                <span>
                  AI STATUS
                </span>

                <strong
                  className={
                    cameraAnalyzing
                      ? "camera-scanning"
                      : "camera-ready"
                  }
                >
                  {cameraAnalyzing
                    ? "SCANNING"
                    : "READY"}
                </strong>

              </div>

            </div>


            {/* CAMERA ERROR */}

            {cameraError && (

              <div className="error-box">
                {cameraError}
              </div>

            )}


            {/* STOP BUTTON */}

            <button
              type="button"
              className="camera-stop-button"
              onClick={stopCamera}
            >
              Stop Camera
            </button>


            {/* HIDDEN CANVAS */}

            <canvas
              ref={canvasRef}
              style={{
                display: "none",
              }}
            />

          </div>

        </div>

      )}

    </div>
  );
}

export default App;