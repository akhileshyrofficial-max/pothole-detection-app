# 🕳️ Pothole Detection App

An AI-powered pothole detection system using YOLOv8, FastAPI (backend), and Streamlit (frontend).

## 🚀 Features
- Upload road images to detect potholes
- API built with FastAPI
- Streamlit dashboard for visualization
- Ready for mobile/edge deployment (ONNX/TFLite export supported)

## 🛠️ Setup

### Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
