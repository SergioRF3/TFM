import io
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential

app = FastAPI(
    title="Fashion Recommendation API",
    description="API para clasificación de prendas de vestir y recuperación de recomendaciones de ropa."
)

# --- 1. CARGA DEL MODELO PYTORCH ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.convnext_tiny(weights=None)
in_features = model.classifier[2].in_features
model.classifier[2] = nn.Linear(in_features, 50)

MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "/tmp/best_convnext_categories.pth")
CONTAINER_NAME = os.getenv("AZURE_STORAGE_CONTAINER_NAME", "tfm-container")
BLOB_NAME = os.getenv("MODEL_BLOB_NAME", "models/best_convnext_latest.pth")
STORAGE_ACCOUNT_NAME = os.getenv("AZURE_STORAGE_ACCOUNT_NAME", "tfmstorage2026")

model_loaded = False

try:
    account_url = f"https://{STORAGE_ACCOUNT_NAME}.blob.core.windows.net"
    credential = DefaultAzureCredential()
    blob_service_client = BlobServiceClient(account_url=account_url, credential=credential)
    blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=BLOB_NAME)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        f.write(blob_client.download_blob().readall())
    print("✅ Modelo descargado desde Azure Blob Storage.")

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    model_loaded = True
    print("✅ Modelo de PyTorch cargado correctamente.")
except Exception as e:
    print(f"⚠️ Error al descargar o cargar el modelo: {e}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

CATEGORY_NAMES = {
    0: "Anorak", 1: "Blazer", 2: "Blouse", 3: "Bomber", 4: "Button-Down",
}


# --- 2. ENDPOINTS DE LA API ---

@app.get("/")
def health_check():
    return {
        "status": "ok",
        "service": "Fashion API backend",
        "model_loaded": model_loaded
    }


@app.post("/predict-image/")
async def predict_image(file: UploadFile = File(...)):
    if not model_loaded:
        raise HTTPException(status_code=503, detail="El modelo no está cargado en el servidor.")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo enviado no es una imagen válida.")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1)
            top_probs, top_indices = torch.topk(probs, k=3)

        predictions = []
        for i in range(3):
            idx = top_indices[0][i].item()
            predictions.append({
                "category_id": idx,
                "category_name": CATEGORY_NAMES.get(idx, str(idx)),
                "confidence": round(top_probs[0][i].item() * 100, 2)
            })

        return {
            "success": True,
            "top_prediction": predictions[0],
            "top_3_predictions": predictions
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar la imagen: {str(e)}")


@app.post("/process-audio-recommendation/")
async def process_audio_recommendation(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="El archivo enviado no es un audio válido.")

    _ = await file.read()

    return {
        "success": True,
        "message": "Audio recibido correctamente.",
        "recommended_outfits": []
    }