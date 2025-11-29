# 🧠 NeuroLoft API: Neural Interface

> *"Exposing Intelligence via RESTful Synapses."*

This project is a production-ready **Microservice** for deploying Deep Learning models. It wraps a PyTorch CNN trained on MNIST into a high-performance **FastAPI** application, containerized with **Docker**.

## 🏗️ Tech Stack
*   **Core**: Python 3.10, PyTorch
*   **API**: FastAPI, Uvicorn
*   **Containerization**: Docker, Docker Compose
*   **Processing**: Pillow, NumPy

## 🚀 Quick Start

### Option A: Docker (Recommended)
Run the entire stack with a single command. The API will be available at `http://localhost:8000`.

```bash
docker-compose up --build
```

### Option B: Local Development
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Train the model (downloads MNIST and saves weights):
   ```bash
   python train.py
   ```
3. Start the server:
   ```bash
   uvicorn app.main:app --reload
   ```

## 🔌 API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Health check & System Status |
| `POST` | `/predict` | Upload an image (PNG/JPG) to get digit classification |

### Example Request
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@digit.png;type=image/png'
```

## 📂 Project Structure
```
NeuroLoft_API/
├── app/
│   ├── artifacts/       # Saved Model Weights (.pth)
│   ├── main.py          # FastAPI Application
│   └── model.py         # PyTorch CNN Architecture
├── Dockerfile           # Container Definition
├── docker-compose.yml   # Orchestration
├── requirements.txt     # Dependencies
├── train.py             # Training Script
└── test_api.py          # Automated API Testing
```
