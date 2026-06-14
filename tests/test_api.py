import base64
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)
VALID_IMAGE_BYTES = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82'

def test_1_health_status():
    assert client.get("/health").status_code == 200

def test_2_health_payload():
    data = client.get("/health").json()
    assert data["status"] == "ok" and data["model_loaded"] is True

def test_3_swagger_docs():
    assert client.get("/docs").status_code == 200

def test_4_openapi_schema():
    assert client.get("/openapi.json").status_code == 200

def test_5_predict_no_payload():
    assert client.post("/predict").status_code == 422

def test_6_predict_bad_extension():
    res = client.post("/predict", files={"file": ("t.txt", b"text", "text/plain")})
    assert res.status_code == 400

def test_7_predict_valid_image_status():
    res = client.post("/predict", files={"file": ("img.png", VALID_IMAGE_BYTES, "image/png")})
    assert res.status_code == 200

def test_8_predict_payload_structure():
    data = client.post("/predict", files={"file": ("img.png", VALID_IMAGE_BYTES, "image/png")}).json()
    assert "pathology_class" in data and "confidence" in data and "grad_cam_base64" in data

def test_9_predict_confidence_range():
    conf = client.post("/predict", files={"file": ("img.png", VALID_IMAGE_BYTES, "image/png")}).json()["confidence"]
    assert 0.0 <= conf <= 1.0

def test_10_predict_valid_b64_canvas():
    b64 = client.post("/predict", files={"file": ("img.png", VALID_IMAGE_BYTES, "image/png")}).json()["grad_cam_base64"]
    assert len(base64.b64decode(b64)) > 0
