
from flask import Flask, request, jsonify
from pathlib import Path
from PIL import Image
from io import BytesIO
from flask_cors import CORS
from dotenv import load_dotenv
from google import genai

import sys
import base64
import os


# --------------------------------------------------
# PATH CONFIGURATION
# --------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent

sys.path.insert(0, str(BACKEND_DIR))

from gradcam_service import generate_gradcam


# --------------------------------------------------
# ENVIRONMENT CONFIGURATION
# --------------------------------------------------

load_dotenv(PROJECT_ROOT / ".env")

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY is missing from .env"
    )

client = genai.Client(api_key=api_key)


# --------------------------------------------------
# FLASK APPLICATION
# --------------------------------------------------

app = Flask(__name__)

CORS(app)


# --------------------------------------------------
# HELPER FUNCTION
# --------------------------------------------------

def image_to_base64(image):
    buffer = BytesIO()

    image.save(buffer, format="PNG")

    return base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")


# --------------------------------------------------
# HOME ROUTE
# --------------------------------------------------

@app.route("/")
def home():
    return jsonify({
        "message": "FedLung-XAI Backend is Running!",
        "status": "success"
    })


# --------------------------------------------------
# ML PREDICTION ENDPOINT
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({
            "error": "No image uploaded"
        }), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({
            "error": "No selected image"
        }), 400

    try:
        image = Image.open(
            file.stream
        ).convert("RGB")

        # Generate prediction and Grad-CAM
        (
            overlay,
            disease,
            confidence,
            region_info,
            explanation
        ) = generate_gradcam(image)

        # Convert images to Base64
        original_base64 = image_to_base64(
            image
        )

        overlay_base64 = image_to_base64(
            overlay
        )

        return jsonify({
            "success": True,
            "predicted_disease": disease,
            "confidence": round(
                confidence, 2
            ),
            "original_image": original_base64,
            "gradcam_overlay": overlay_base64,
            "region_info": region_info,
            "explanation": explanation,
            "disclaimer": (
                "AI prediction—not a medical diagnosis"
            )
        })

    except Exception as error:

        print("Prediction error:", error)

        return jsonify({
            "success": False,
            "error": str(error)
        }), 500


# --------------------------------------------------
# GEMINI CHAT ENDPOINT
# --------------------------------------------------

@app.route("/chat", methods=["POST"])
def chat():

    try:
        data = request.get_json()

        if not data or "message" not in data:
            return jsonify({
                "error": "Message is required"
            }), 400

        user_message = data["message"].strip()

        if not user_message:
            return jsonify({
                "error": "Message cannot be empty"
            }), 400

        prediction = data.get("prediction")

        if prediction:

            prediction_context = f"""
Current model prediction:

- Predicted disease:
{prediction.get("predicted_disease", "Unknown")}

- Confidence:
{prediction.get("confidence", "Unknown")}%

- Region information:
{prediction.get("region_info", "Unavailable")}

- Model explanation:
{prediction.get("explanation", "Unavailable")}
"""

        else:

            prediction_context = """
No chest X-ray prediction is currently available.
"""

        prompt = f"""
You are the Gemini AI assistant inside the
FedLung-XAI research application.

{prediction_context}

User question:
{user_message}

Instructions:

1. Use prediction information only as AI model output.
2. Do not claim that the prediction is medically confirmed.
3. Explain the model output in simple language when relevant.
4. Do not provide a medical diagnosis.
5. Do not claim that Grad-CAM proves a disease or confirms lesions.
6. Recommend consultation with a qualified healthcare professional
   for medical concerns.
7. If the question is unrelated to the prediction, answer normally.
8. Be clear, helpful, and concise.
9. Answer like a natural WhatsApp conversation.
10. Keep normal answers short, usually 2–5 sentences.
11. Avoid 8-mark exam-style answers, long headings,
    unnecessary numbered lists, and repeated disclaimers.
12. Expand only when the user specifically asks for detail.
"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        return jsonify({
            "reply": response.text
        })

    except Exception as error:

        print("Chat error:", error)

        return jsonify({
            "error": str(error)
        }), 500


# --------------------------------------------------
# START FLASK SERVER
# --------------------------------------------------

if __name__ == "__main__":

    print("Starting FedLung-XAI backend...")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
