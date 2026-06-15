import io
import base64
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
import torchvision.transforms as transforms
from PIL import Image

# Импортируем нашу модель и Grad-CAM из соседних файлов
from model import MedicalCNN
from grad_cam import run_grad_cam

app = FastAPI(
    title="Medical Image Classifier API",
    version="0.1.0",
    description="Сервис классификации медицинских снимков с объяснимостью Grad-CAM"
)

# Глобальные переменные для модели
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
CLASSES = ["Normal", "Pneumonia"]

@app.on_event("startup")
async def load_model():
    global model
    # Инициализируем архитектуру и загружаем веса (или mock-веса для демонстрации)
    model = MedicalCNN()
    # torch.load("weights.pth", map_location=device) # Раскомментировать при наличии реальных весов
    model.to(device)
    model.eval()

@app.get("/health", summary="Health Check")
async def health_check():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict", summary="Predict Image")
async def predict(file: UploadFile = File(...)):
    # Проверяем формат файла
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Разрешены только изображения JPEG и PNG.")
    
    try:
        # Читаем байты загруженного файла динамически (исправляет баг "застывшего" ответа)
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Предобработка для PyTorch модели
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # 1. Получаем предсказание модели
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, class_idx = torch.max(probabilities, dim=1)
            
        pathology_class = CLASSES[class_idx.item()]
        conf_score = round(confidence.item(), 2)
        
        # 2. Генерируем карту объяснимости Grad-CAM для целевого класса
        grad_cam_img = run_grad_cam(model, input_tensor, class_idx.item(), image)
        
        # Кодируем полученное изображение в Base64 для передачи по API
        buffered = io.BytesIO()
        grad_cam_img.save(buffered, format="JPEG")
        grad_cam_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return {
            "pathology_class": pathology_class,
            "confidence": conf_score,
            "grad_cam_base64": grad_cam_base64
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке снимка: {str(e)}")