from fastapi import FastAPI, File, UploadFile, HTTPException
from app.schemas import PredictionResponse, HealthCheck
from app.services.ml_engine import MedicalModelService
import cv2
import numpy as np

app = FastAPI(
    title="Medical Image Classifier API",
    description="Сервис классификации медицинских снимков с объяснимостью Grad-CAM",
    version="0.1.0"
)

ml_service = MedicalModelService()

@app.get("/health", response_model=HealthCheck, tags=["System"])
async def health_check():
    return {"status": "ok", "model_loaded": ml_service.is_loaded}

@app.post("/predict", response_model=PredictionResponse, tags=["ML"])
async def predict_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Файл должен быть изображением")
    
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        result = ml_service.predict(img)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")
