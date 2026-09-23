
import { useState, useRef, useEffect } from 'react'
import './ChatPanel.css'

const CHAT_API_URL = 'http://127.0.0.1:5000/chat'

function ChatPanel({ onClose, prediction }) {
  const [message, setMessage] = useState('')
  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(false)

  const messagesEndRef = useRef(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({
      behavior: 'smooth',
    })
  }, [messages, loading])

  const sendMessage = async (event) => {
    event.preventDefault()

    const userMessage = message.trim()

    if (!userMessage || loading) return

    setMessages((previous) => [
      ...previous,
      { role: 'user', text: userMessage },
    ])

    setMessage('')
    setLoading(true)

    try {
      const response = await fetch(CHAT_API_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: userMessage,
          prediction: prediction
            ? {
                predicted_disease: prediction.predicted_disease,
                confidence: prediction.confidence,
                region_info: prediction.region_info,
                explanation: prediction.explanation,
              }
            : null,
        }),
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.error || 'Chat request failed')
      }

      setMessages((previous) => [
        ...previous,
        { role: 'assistant', text: data.reply },
      ])
    } catch (error) {
      setMessages((previous) => [
        ...previous,
        { role: 'assistant', text: `Error: ${error.message}` },
      ])
    } finally {
      setLoading(false)
    }
  }

  return (
    <aside className="chat-panel">

      {/* HEADER */}
      <div className="chat-header">

        <div className="chat-header-info">
          <div className="chat-ai-icon">✦</div>

          <div>
            <h3>Gemini AI</h3>
            <p>FedLung-XAI Assistant</p>
          </div>
        </div>

        <button
          className="chat-close-button"
          onClick={onClose}
          aria-label="Close chat"
        >
          ×
        </button>

      </div>

      {/* PREDICTION CONTEXT */}
      {prediction && (
        <div className="chat-prediction-context">
          <span>Current Prediction</span>

          <strong>
            {prediction.predicted_disease}
          </strong>

          <small>
            Confidence: {prediction.confidence}%
          </small>
        </div>
      )}

      {/* MESSAGES */}
      <div className="chat-messages">

        {messages.length === 0 && (
          <div className="chat-welcome">
            <div className="welcome-icon">✦</div>

            <h4>Hey! 👋</h4>

            <p>
              Ask me anything about your X-ray prediction.
              I can help explain the model's output.
            </p>

            <small>
              AI-generated information. Not a medical diagnosis.
            </small>
          </div>
        )}

        {messages.map((item, index) => (
          <div
            key={index}
            className={`chat-message-row ${item.role}`}
          >
            <div className="chat-avatar">
              {item.role === 'user' ? 'You' : 'AI'}
            </div>

            <div className={`chat-message ${item.role}`}>
              <p>{item.text}</p>
            </div>
          </div>
        ))}

        {loading && (
          <div className="chat-message-row assistant">
            <div className="chat-avatar">AI</div>

            <div className="chat-message assistant typing-message">
              <p>Typing...</p>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />

      </div>

      {/* INPUT */}
      <form
        className="chat-input-area"
        onSubmit={sendMessage}
      >
        <input
          type="text"
          placeholder="Message Gemini..."
          value={message}
          onChange={(event) => setMessage(event.target.value)}
          disabled={loading}
          aria-label="Message Gemini"
        />

        <button
          type="submit"
          disabled={loading || !message.trim()}
          aria-label="Send message"
        >
          ↑
        </button>
      </form>

      {/* FOOTER */}
      <div className="chat-footer">
        AI support • Verify medical information
      </div>

    </aside>
  )
}

export default ChatPanel
