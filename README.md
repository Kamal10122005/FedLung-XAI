# FedLung-XAI

## Privacy-Preserving Federated Multi-Class Lung Disease Diagnosis with Explainable AI

FedLung-XAI is an AI-based lung disease diagnosis system that combines Federated Learning with Explainable AI (XAI) for chest X-ray classification.

The system uses a multi-class deep learning model to classify chest X-ray images into four disease categories and uses Grad-CAM to provide visual explanations for model predictions.

> **Disclaimer:** This project is intended for research and educational purposes only. AI predictions are not medical diagnoses.

---

## Features

- Multi-class lung disease classification
- Four supported classes:
  - COVID-19
  - NORMAL
  - PNEUMONIA
  - TUBERCULOSIS
- Federated Learning using Flower
- Grad-CAM based Explainable AI
- Prediction confidence scores
- Original X-ray visualization
- Grad-CAM heatmap visualization
- React-based web interface
- Flask backend
- AI chat assistant
- Training and evaluation scripts

---

## Project Structure

```text
FedLung-XAI/
├── backend/
│   ├── app.py
│   ├── gradcam_service.py
│   └── ...
│
├── frontend/
│   └── src/
│       ├── components/
│       │   ├── ChatPanel.jsx
│       │   └── ChatPanel.css
│       ├── App.jsx
│       └── App.css
│
├── src/
│   ├── data_loader.py
│   ├── multidisease_loader.py
│   ├── multidisease_model.py
│   ├── train.py
│   ├── train_multidisease.py
│   ├── evaluate.py
│   ├── evaluate_multidisease.py
│   ├── gradcam.py
│   ├── federated_data.py
│   ├── fl_client.py
│   ├── fl_server.py
│   └── run_federated.py
│
├── data/
│   ├── multidisease/
│   └── chest_xray/
│
├── requirements.txt
└── README.md