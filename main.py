import io
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="Fashion Recommendation API",
    description="API para clasificación de prendas de vestir y recuperación de recomendaciones de ropa."
)

# --- 1. CARGA DEL MODELO PYTORCH ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instanciamos la arquitectura ConvNeXt-Tiny
model = models.convnext_tiny(weights=None)
in_features = model.classifier[2].in_features
model.classifier[2] = nn.Linear(in_features, 50)

# Cargar pesos guardados
MODEL_PATH = "best_convnext_categories.pth"
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print("✅ Modelo de PyTorch cargado correctamente.")
except Exception as e:
    print(f"⚠️ No se pudo cargar el modelo: {e}")

# Transformación estándar para la inferencia
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Mapeo de categorías (Ajusta según tu df_cat)
CATEGORY_NAMES = {
    0: "Anorak", 1: "Blazer", 2: "Blouse", 3: "Bomber", 4: "Button-Down",
    # Add remaining category mappings here...
}


# --- 2. ENDPOINTS DE LA API ---

@app.get("/")
def health_check():
    return {"status": "ok", "service": "Fashion API backend"}


@app.post("/predict-image/")
async def predict_image(file: UploadFile = File(...)):
    """
    Recibe una imagen (JPG/PNG), la procesa con ConvNeXt-Tiny
    y devuelve la categoría predicha con sus probabilidades.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo enviado no es una imagen válida.")

    try:
        # Leer la imagen enviada
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Preprocesar
        tensor = transform(image).unsqueeze(0).to(device)
        
        # Inferencia
        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1)
            top_probs, top_indices = torch.topk(probs, k=3)
            
        # Formatear respuesta
        predictions = []
        for i in range(3):
            idx = top_indices[0][i].item()
            predictions.append({
                "category_id": idx,
                "confidence": round(top_probs[0][i].item() * 100, 2)
            })
            
        return {
            "success": True,
            "top_prediction": predictions[0],
            "top_3_predictions": predictions
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar la imagen: {str(e)}")


@app.post("/process-audio-recommendation/")
async def process_audio_recommendation(file: UploadFile = File(...)):
    """
    Endpoint para recibir notas de audio (e.g., MP3/WAV/M4A),
    transcribir/analizar la petición y devolver combinaciones de ropa.
    """
    if not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="El archivo enviado no es un audio válido.")
    
    # 1. Guardar o leer los bytes del audio
    audio_bytes = await file.read()
    
    # TODO: Integrar aquí la transcripción (ej. Whisper o Gemini Audio) 
    # y el motor de recomendación de combinaciones de ropa.
    
    return {
        "success": True,
        "message": "Audio recibido correctamente.",
        "recommended_outfits": []
    }