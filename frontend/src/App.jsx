import { useEffect, useRef, useState } from "react";
import "./App.css";

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
  const lastFrameTimeRef = useRef(0);
  const frameBusyRef = useRef(false);


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
  // CLEAN CAMERA WHEN PAGE CLOSES
  // =====================================================

  useEffect(() => {

    return () => {
      stopCamera();
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

    // Same image can be selected again
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
        "http://127.0.0.1:8000/predict",
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
  // START CAMERA
  // =====================================================

  const startCamera = async () => {

    try {

      setCameraError("");
      setCameraPredictions([]);
      setCameraFaces(0);
      setFps(0);


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
              ideal: 640
            },
            height: {
              ideal: 480
            }
          },
          audio: false
        });


      streamRef.current = stream;


      if (videoRef.current) {

        videoRef.current.srcObject =
          stream;

        await videoRef.current.play();
      }


      setCameraActive(true);

      // Start frame analysis
      startFrameLoop();

    } catch (err) {

      console.error(err);

      setCameraError(
        err.message ||
        "Could not access your camera."
      );

      setCameraActive(false);
    }
  };


  // =====================================================
  // STOP CAMERA
  // =====================================================

  const stopCamera = () => {

    // Stop timer
    if (cameraTimerRef.current) {

      clearTimeout(
        cameraTimerRef.current
      );

      cameraTimerRef.current = null;
    }


    // Stop camera stream
    if (streamRef.current) {

      streamRef.current
        .getTracks()
        .forEach((track) => {
          track.stop();
        });

      streamRef.current = null;
    }


    if (videoRef.current) {

      videoRef.current.srcObject = null;
    }


    frameBusyRef.current = false;

    setCameraActive(false);
    setCameraAnalyzing(false);
    setCameraPredictions([]);
    setCameraFaces(0);
    setFps(0);
  };


  // =====================================================
  // CAMERA FRAME LOOP
  // =====================================================

  const startFrameLoop = () => {

    const processFrame = async () => {

      if (!streamRef.current) {
        return;
      }


      // Prevent multiple requests at once
      if (frameBusyRef.current) {

        cameraTimerRef.current =
          setTimeout(
            processFrame,
            150
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
        video.videoWidth === 0
      ) {

        cameraTimerRef.current =
          setTimeout(
            processFrame,
            200
          );

        return;
      }


      frameBusyRef.current = true;
      setCameraAnalyzing(true);


      try {

        const width =
          video.videoWidth;

        const height =
          video.videoHeight;


        canvas.width = width;
        canvas.height = height;


        const context =
          canvas.getContext("2d");


        context.drawImage(
          video,
          0,
          0,
          width,
          height
        );


        const blob =
          await new Promise(
            (resolve) =>
              canvas.toBlob(
                resolve,
                "image/jpeg",
                0.65
              )
          );


        if (!blob) {
          throw new Error(
            "Could not capture camera frame."
          );
        }


        const formData =
          new FormData();


        formData.append(
          "file",
          blob,
          "camera-frame.jpg"
        );


        const startTime =
          performance.now();


        const response =
          await fetch(
            "http://127.0.0.1:8000/predict-frame",
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
            "Camera analysis failed."
          );
        }


        setCameraPredictions(
          data.predictions || []
        );

        setCameraFaces(
          data.faces_detected || 0
        );


        // ------------------------------------------------
        // FPS
        // ------------------------------------------------

        const endTime =
          performance.now();

        const processingTime =
          endTime - startTime;


        if (processingTime > 0) {

          const currentFps =
            Math.min(
              30,
              Math.round(
                1000 /
                processingTime
              )
            );

          setFps(currentFps);
        }


      } catch (err) {

        console.error(
          "Camera analysis error:",
          err
        );

        setCameraError(
          err.message ||
          "Camera analysis failed."
        );

      } finally {

        frameBusyRef.current = false;
        setCameraAnalyzing(false);


        // About 4-5 analysis requests/sec maximum
        if (streamRef.current) {

          cameraTimerRef.current =
            setTimeout(
              processFrame,
              180
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
            CAMERA SECTION
            ================================================= */}

        <section
          className="upload-card"
          style={{
            marginTop: "28px"
          }}
        >

          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              gap: "20px",
              marginBottom: "20px",
              flexWrap: "wrap"
            }}
          >

            <div>

              <h2
                style={{
                  margin: 0,
                  fontSize: "26px",
                  fontWeight: 800
                }}
              >
                Live Camera Detection
              </h2>

              <p
                style={{
                  marginTop: "8px",
                  color: "#8290aa",
                  fontSize: "14px"
                }}
              >
                Detect multiple faces in real time.
              </p>

            </div>


            {cameraActive && (

              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "12px"
                }}
              >

                <span
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: "7px",
                    color: "#35e890",
                    fontSize: "13px",
                    fontWeight: 700
                  }}
                >

                  <span
                    style={{
                      width: "8px",
                      height: "8px",
                      borderRadius: "50%",
                      background: "#35e890",
                      boxShadow:
                        "0 0 12px rgba(53,232,144,.8)"
                    }}
                  />

                  CAMERA LIVE

                </span>


                <span
                  style={{
                    color: "#9aa8c2",
                    fontSize: "13px",
                    fontWeight: 700
                  }}
                >
                  {fps} FPS
                </span>

              </div>

            )}

          </div>


          {/* CAMERA VIEW */}

          <div
            style={{
              position: "relative",
              width: "100%",
              aspectRatio: "16 / 9",
              minHeight: "300px",
              overflow: "hidden",
              borderRadius: "22px",
              background:
                "rgba(4,8,18,.9)",
              border:
                "1px solid rgba(120,155,220,.16)"
            }}
          >

            <video
              ref={videoRef}
              muted
              playsInline
              style={{
                width: "100%",
                height: "100%",
                objectFit: "cover",
                display:
                  cameraActive
                    ? "block"
                    : "none",
                transform: "scaleX(-1)"
              }}
            />


            {!cameraActive && (

              <div
                style={{
                  position: "absolute",
                  inset: 0,
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  justifyContent: "center",
                  textAlign: "center",
                  padding: "30px"
                }}
              >

                <div
                  style={{
                    fontSize: "48px",
                    marginBottom: "14px"
                  }}
                >
                  ◉
                </div>


                <h3
                  style={{
                    fontSize: "20px",
                    marginBottom: "8px"
                  }}
                >
                  Camera is ready
                </h3>


                <p
                  style={{
                    color: "#8290aa",
                    fontSize: "14px"
                  }}
                >
                  Start your camera to begin
                  real-time detection.
                </p>

              </div>

            )}


            {/* FACE BOXES */}

            {cameraActive &&
              cameraPredictions.map(
                (prediction, index) => {

                  const box =
                    prediction.bounding_box;

                  if (!box) return null;


                  const video =
                    videoRef.current;


                  if (!video) return null;


                  const videoWidth =
                    video.videoWidth || 640;

                  const videoHeight =
                    video.videoHeight || 480;


                  const left =
                    (box.x / videoWidth) *
                    100;

                  const top =
                    (box.y / videoHeight) *
                    100;

                  const width =
                    (box.width / videoWidth) *
                    100;

                  const height =
                    (box.height / videoHeight) *
                    100;


                  const isFake =
                    prediction.result ===
                    "FAKE";


                  return (

                    <div
                      key={`${prediction.face}-${index}`}
                      style={{
                        position: "absolute",

                        left: `${100 - left - width}%`,
                        top: `${top}%`,

                        width: `${width}%`,
                        height: `${height}%`,

                        border:
                          `2px solid ${
                            isFake
                              ? "#ff5470"
                              : "#35e890"
                          }`,

                        borderRadius: "10px",

                        boxShadow:
                          isFake
                            ? "0 0 18px rgba(255,84,112,.45)"
                            : "0 0 18px rgba(53,232,144,.35)",

                        pointerEvents: "none"
                      }}
                    >

                      <div
                        style={{
                          position: "absolute",
                          top: "-34px",
                          left: "0",

                          padding:
                            "6px 10px",

                          borderRadius:
                            "8px",

                          background:
                            isFake
                              ? "rgba(255,84,112,.92)"
                              : "rgba(53,232,144,.92)",

                          color: "#04100a",

                          fontSize: "12px",
                          fontWeight: 800,

                          whiteSpace: "nowrap"
                        }}
                      >

                        Face {prediction.face}
                        {" · "}
                        {prediction.result}
                        {" · "}
                        {prediction.confidence}%

                      </div>

                    </div>

                  );
                }
              )}

          </div>


          {/* CAMERA CONTROLS */}

          <div
            style={{
              display: "flex",
              gap: "12px",
              marginTop: "18px",
              flexWrap: "wrap"
            }}
          >

            {!cameraActive ? (

              <button
                type="button"
                className="analyze-button"
                style={{
                  marginTop: 0,
                  flex: 1
                }}
                onClick={startCamera}
              >
                Start Camera
              </button>

            ) : (

              <button
                type="button"
                className="analyze-button"
                style={{
                  marginTop: 0,
                  flex: 1,
                  background:
                    "linear-gradient(100deg,#ff405f,#ff687d)"
                }}
                onClick={stopCamera}
              >
                Stop Camera
              </button>

            )}

          </div>


          {/* CAMERA STATUS */}

          {cameraActive && (

            <div
              style={{
                display: "grid",
                gridTemplateColumns:
                  "repeat(auto-fit,minmax(150px,1fr))",
                gap: "12px",
                marginTop: "16px"
              }}
            >

              <div
                className="result-item"
                style={{
                  minHeight: "70px"
                }}
              >

                <div>

                  <span>
                    Faces Detected
                  </span>

                  <strong>
                    {cameraFaces}
                  </strong>

                </div>

              </div>


              <div
                className="result-item"
                style={{
                  minHeight: "70px"
                }}
              >

                <div>

                  <span>
                    Analysis
                  </span>

                  <strong
                    style={{
                      color:
                        cameraAnalyzing
                          ? "#68a7ff"
                          : "#35e890"
                    }}
                  >
                    {cameraAnalyzing
                      ? "Scanning"
                      : "Ready"}
                  </strong>

                </div>

              </div>

            </div>

          )}


          {cameraError && (

            <div className="error-box">
              {cameraError}
            </div>

          )}


          {/* Hidden canvas used to capture frames */}

          <canvas
            ref={canvasRef}
            style={{
              display: "none"
            }}
          />

        </section>


        {/* =================================================
            IMAGE RESULT
            ================================================= */}

        {result && (

          <section className="result-card">

            <h2>
              Analysis Result
            </h2>


            <div className="result-info">

              <div className="result-item">

                <div>

                  <span>
                    Faces Detected
                  </span>

                  <strong>
                    {result.faces_detected}
                  </strong>

                </div>

              </div>


              {result.predictions &&
                result.predictions.map(
                  (prediction, index) => (

                    <div
                      className="prediction"
                      key={index}
                    >

                      <div>

                        <span>
                          Face {prediction.face || index + 1}
                        </span>

                        <strong
                          className={
                            prediction.result ===
                            "FAKE"
                              ? "fake"
                              : "real"
                          }
                        >
                          {prediction.result}
                        </strong>

                      </div>


                      <div>

                        <span>
                          Confidence
                        </span>

                        <strong>
                          {prediction.confidence}%
                        </strong>

                      </div>

                    </div>

                  )
                )}

            </div>


            {result.message && (

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