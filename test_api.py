import base64
import pytest
from fastapi.testclient import TestClient

# ВАЖНО: Если main.py лежит в корне проекта, пишем так. 
# Если внутри папки app/, то замените на: from app.main import app
from main import app

client = TestClient(app)

# Минимальный валлидный PNG-файл 1x1 пиксель в байтах для тестов
VALID_IMAGE_BYTES = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82'

# --- БЛОК 1: ТЕСТИРОВАНИЕ HEALTH И ДОКУМЕНТАЦИИ ---

def test_1_health_status():
    assert client.get("/health").status_code == 200

def test_2_health_payload():
    data = client.get("/health").json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True

def test_3_swagger_docs():
    assert client.get("/docs").status_code == 200

def test_4_openapi_schema():
    assert client.get("/openapi.json").status_code == 200


# --- БЛОК 2: ВАЛИДАЦИЯ ВХОДНЫХ ДАННЫХ (ОШИБКИ) ---

def test_5_predict_no_payload():
    assert client.post("/predict").status_code == 422

def test_6_predict_bad_extension():
    res = client.post("/predict", files={"file": ("t.txt", b"text", "text/plain")})
    assert res.status_code == 400


# --- БЛОК 3: ИНФЕРЕНС И СТРУКТУРА ОТВЕТА МОДЕЛИ ---

def test_7_predict_valid_image_status():
    res = client.post("/predict", files={"file": ("img.png", VALID_IMAGE_BYTES, "image/png")})
    assert res.status_code == 200

def test_8_predict_payload_structure():
    data = client.post("/predict", files={"file": ("img.png", VALID_IMAGE_BYTES, "image/png")}).json()
    assert "pathology_class" in data
    assert "confidence" in data
    assert "grad_cam_base64" in data

def test_9_predict_confidence_range():
    conf = client.post("/predict", files={"file": ("img.png", VALID_IMAGE_BYTES, "image/png")}).json()["confidence"]
    assert 0.0 <= conf <= 1.0

def test_10_predict_valid_b64_canvas():
    b64 = client.post("/predict", files={"file": ("img.png", VALID_IMAGE_BYTES, "image/png")}).json()["grad_cam_base64"]
    assert len(base64.b64decode(b64)) > 0


# --- БЛОК 4: СТРЕСС-ТЕСТЫ И СТАБИЛЬНОСТЬ ---

def test_11_predict_caching_consistency():
    """Проверяет стабильность кэша при повторной отправке одного и того же файла."""
    payload = {"file": ("img.png", VALID_IMAGE_BYTES, "image/png")}
    res1 = client.post("/predict", files=payload).json()
    res2 = client.post("/predict", files=payload).json()
    assert res1["pathology_class"] == res2["pathology_class"]
    assert res1["confidence"] == res2["confidence"]
    assert res1["grad_cam_base64"] == res2["grad_cam_base64"]

def test_12_predict_corrupted_image():
    """Проверяет реакцию сервиса на поврежденное изображение."""
    res = client.post("/predict", files={"file": ("corrupted.png", b"not_real_image_bytes", "image/png")})
    assert res.status_code == 500
    assert "Ошибка при обработке снимка" in res.json()["detail"]