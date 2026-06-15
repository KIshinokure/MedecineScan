import base64
import pytest
import io
from PIL import Image
from fastapi.testclient import TestClient
from main import app

img = Image.new("RGB", (100, 100), color="white")
img_byte_arr = io.BytesIO()
img.save(img_byte_arr, format="PNG")
VALID_IMAGE_BYTES = img_byte_arr.getvalue()

@pytest.fixture(scope="module")
def test_client():
    with TestClient(app) as client:
        yield client

def test_1_health_status(test_client):
    assert test_client.get("/health").status_code == 200

def test_2_health_payload(test_client):
    data = test_client.get("/health").json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True

def test_3_swagger_docs(test_client):
    assert test_client.get("/docs").status_code == 200

def test_4_openapi_schema(test_client):
    assert test_client.get("/openapi.json").status_code == 200

def test_5_predict_no_payload(test_client):
    assert test_client.post("/predict").status_code == 422

def test_6_predict_bad_extension(test_client):
    res = test_client.post("/predict", files={"file": ("t.txt", b"text", "text/plain")})
    assert res.status_code == 400

def test_7_predict_valid_image_status(test_client):
    res = test_client.post("/predict", files={"file": ("img.png", VALID_IMAGE_BYTES, "image/png")})
    assert res.status_code == 200

def test_8_predict_payload_structure(test_client):
    data = test_client.post("/predict", files={"file": ("img.png", VALID_IMAGE_BYTES, "image/png")}).json()
    assert "pathology_class" in data
    assert "confidence" in data
    assert "grad_cam_base64" in data

def test_9_predict_confidence_range(test_client):
    conf = test_client.post("/predict", files={"file": ("img.png", VALID_IMAGE_BYTES, "image/png")}).json()["confidence"]
    assert 0.0 <= conf <= 1.0

def test_10_predict_valid_b64_canvas(test_client):
    b64 = test_client.post("/predict", files={"file": ("img.png", VALID_IMAGE_BYTES, "image/png")}).json()["grad_cam_base64"]
    assert len(base64.b64decode(b64)) > 0

def test_11_predict_caching_consistency(test_client):
    payload = {"file": ("img.png", VALID_IMAGE_BYTES, "image/png")}
    res1 = test_client.post("/predict", files=payload).json()
    res2 = test_client.post("/predict", files=payload).json()
    assert res1["pathology_class"] == res2["pathology_class"]
    assert res1["confidence"] == res2["confidence"]
    assert res1["grad_cam_base64"] == res2["grad_cam_base64"]

def test_12_predict_corrupted_image(test_client):
    res = test_client.post("/predict", files={"file": ("corrupted.png", b"not_real_image_bytes", "image/png")})
    assert res.status_code == 500
    assert "Ошибка при обработке снимка" in res.json()["detail"]
