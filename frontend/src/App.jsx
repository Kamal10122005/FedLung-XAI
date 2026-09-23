
import { useState } from 'react'
import './App.css'
import ChatPanel from './components/ChatPanel'

const API_URL = 'http://127.0.0.1:5000/predict'

function App() {
  const [preview, setPreview] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [showOverlay, setShowOverlay] = useState(true)
  const [error, setError] = useState(null)
  const [showChat, setShowChat] = useState(false)

  const handleImageUpload = async (event) => {
    const file = event.target.files[0]

    if (!file) return

    setPreview(URL.createObjectURL(file))
    setResult(null)
    setError(null)
    setShowOverlay(true)
    setLoading(true)

    const formData = new FormData()
    formData.append('image', file)

    try {
      const response = await fetch(API_URL, {
        method: 'POST',
        body: formData,
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.error || 'Prediction failed')
      }

      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const overlayImage = result
    ? `data:image/png;base64,${result.gradcam_overlay}`
    : null

  const getMainExplanation = (explanation) => {
    if (!explanation) return ''

    return explanation
      .split('\n')
      .filter(
        (line) =>
          !line.includes('Attention Region') &&
          !line.includes('Activation Strength') &&
          !line.includes('Attention Coverage')
      )
      .join('\n')
  }

  return (
    <div className="app">

      {/* HEADER */}
      <header className="topbar">

        {/* AI TOGGLE */}
        <button
          className={`ai-toggle ${showChat ? 'active' : ''}`}
          onClick={() => setShowChat(!showChat)}
          aria-label="Toggle AI chat"
        >
          <span className="toggle-knob">
            {showChat && 'AI'}
          </span>
        </button>

        {/* CENTER BRAND */}
        <div className="brand">
          <h2>FedLung-XAI</h2>
        </div>

        {/* UPLOAD BUTTON */}
        <div className="header-actions">
          <label className="upload-button">
            Upload X-ray

            <input
              type="file"
              accept="image/*"
              onChange={handleImageUpload}
              hidden
            />
          </label>
        </div>

      </header>

      {/* THREE-COLUMN DASHBOARD */}
      <main className="dashboard">

        {/* COLUMN 1: GRAD-CAM */}
        <section className="result-card gradcam-card">

          <h3>Grad-CAM Visualization</h3>

          <div className="gradcam-content">

            {overlayImage ? (
              <>
                <img
                  src={showOverlay ? overlayImage : preview}
                  alt={
                    showOverlay
                      ? 'Grad-CAM heatmap overlay'
                      : 'Original chest X-ray'
                  }
                />

                <div className="attention-scale">
                  <span>High Attention</span>
                  <div className="color-scale"></div>
                  <span>Low Attention</span>
                </div>
              </>
            ) : (
              <div className="gradcam-empty">
                <span>◎</span>
                <p>Grad-CAM visualization will appear here.</p>
              </div>
            )}

          </div>

          <button
            className="overlay-button"
            onClick={() => setShowOverlay(!showOverlay)}
            disabled={!result}
          >
            {showOverlay ? 'Show Original View' : 'Show Grad-CAM'}
          </button>

        </section>

        {/* COLUMN 2: ORIGINAL X-RAY + PREDICTION */}
        <section className="middle-panel">

          {/* ORIGINAL X-RAY */}
          <section className="image-panel">

            <div className="panel-heading">
              <h3>Original X-ray</h3>
            </div>

            <div className="image-container">

              {preview ? (
                <img
                  src={preview}
                  alt="Original chest X-ray"
                />
              ) : (
                <div className="empty-state">
                  <span>◉</span>
                  <p>Upload a chest X-ray to begin</p>

                  <small>
                    Supported formats: JPG, JPEG, PNG
                  </small>
                </div>
              )}

            </div>

            <div className="image-footer">
              <span>Original X-ray Image</span>
            </div>

          </section>

          {/* PREDICTION RESULT */}
          <div className="result-card prediction-card">

            <h3>Prediction Result</h3>

            <div className="prediction-placeholder">

              {loading ? (
                <>
                  <span>◌</span>
                  <h2>Analyzing...</h2>
                  <p>AI model is processing the X-ray.</p>
                </>
              ) : result ? (
                <>
                  <h2>{result.predicted_disease}</h2>

                  <p>
                    Confidence: {result.confidence}%
                  </p>

                  <div className="progress-bar">
                    <div
                      style={{
                        width: `${result.confidence}%`,
                      }}
                    ></div>
                  </div>

                  <strong className="confidence-value">
                    {result.confidence}%
                  </strong>
                </>
              ) : (
                <>
                  <span>—</span>
                  <h2>Awaiting Analysis</h2>

                  <p>
                    Upload an X-ray to receive an AI prediction.
                  </p>
                </>
              )}

            </div>

            <div className="disclaimer">

              <strong>
                ⚠ AI prediction—not a medical diagnosis.
              </strong>

              <p>
                Please consult a qualified healthcare professional.
              </p>

            </div>

            {error && (
              <p className="error-message">
                Error: {error}
              </p>
            )}

          </div>

        </section>

        {/* COLUMN 3: MODEL EXPLANATION */}
        <section className="result-card model-explanation-card">

          <h3>Model Explanation</h3>

          <div className="explanation-boxes">

            {/* ATTENTION REGION */}
            <div className="explanation-box attention-box">

              <div className="explanation-box-title">
                <span>◎</span>
                <h4>Attention Region</h4>
              </div>

              {result?.region_info ? (
                <div className="attention-details">

                  <p>
                    <strong>Region:</strong>{' '}
                    {result.region_info.region}
                  </p>

                  <p>
                    <strong>Activation Strength:</strong>{' '}
                    {result.region_info.activation_strength}
                  </p>

                  <p>
                    <strong>Attention Coverage:</strong>{' '}
                    {result.region_info.attention_percentage}%
                  </p>

                  <p>
                    <strong>Interpretation:</strong>{' '}
                    These areas influenced the model's prediction.
                  </p>

                </div>
              ) : (
                <p>
                  Attention region details will appear
                  after X-ray analysis.
                </p>
              )}

            </div>

            {/* EXPLANATION */}
            <div className="explanation-box explanation-main">

              <div className="explanation-box-title">
                <span>▣</span>
                <h4>Explanation</h4>
              </div>

              {result ? (
                <p>
                  {getMainExplanation(result.explanation)}
                </p>
              ) : (
                <p>
                  Grad-CAM will highlight areas influencing
                  the model's prediction.
                </p>
              )}

            </div>

          </div>

        </section>

      </main>

      {/* FLOATING GEMINI AI CHAT POPUP */}
      {showChat && (
        <div className="chat-popup-wrapper">
          <ChatPanel
            onClose={() => setShowChat(false)}
            prediction={result}
          />
        </div>
      )}

    </div>
  )
}

export default App
