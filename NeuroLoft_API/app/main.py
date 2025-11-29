from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import io
import torch
import torchvision.transforms as transforms
from app.model import NeuroCNN
import torch.nn.functional as F

app = FastAPI(
    title="NeuroLoft Neural API",
    description="Digital Synapse Interface for MNIST Classification",
    version="1.0.0"
)

# --- Global State ---
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Lifecycle: Load Model on Startup ---
@app.on_event("startup")
async def load_model():
    global model
    print(f"🔌 Initializing Neural Interface on {device}...")
    try:
        model = NeuroCNN().to(device)
        # Load weights
        model.load_state_dict(torch.load("app/artifacts/neuro_mnist.pth", map_location=device))
        model.eval()
        print("✅ Neural Core Online. Weights loaded successfully.")
    except Exception as e:
        print(f"❌ Critical Failure: Could not load model. {e}")

# --- Preprocessing Pipeline ---
def transform_image(image_bytes):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0).to(device)

# --- Endpoints ---
@app.get("/")
async def root():
    return {
        "system": "NeuroLoft API",
        "status": "operational",
        "version": "1.0.0"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not model:
        raise HTTPException(status_code=503, detail="Neural Core not initialized")
    
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Only JPEG/PNG allowed.")

    try:
        image_bytes = await file.read()
        tensor = transform_image(image_bytes)
        
        with torch.no_grad():
            outputs = model(tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
        return {
            "digit": int(predicted.item()),
            "confidence": float(confidence.item()),
            "probabilities": probabilities.cpu().numpy().tolist()[0]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Error: {str(e)}")
