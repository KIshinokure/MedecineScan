import torch
import cv2
import numpy as np
import base64
import hashlib

class MedicalModelService:
    def __init__(self):
        self.is_loaded = True
        self.classes = ["Normal", "Pneumonia"]
        self._cache = {}

    def _compute_image_hash(self, image_bgr: np.ndarray) -> str:
        """Генерирует уникальный MD5 хэш для матрицы изображения."""
        if image_bgr is None:
            return ""
        return hashlib.md5(image_bgr.tobytes()).hexdigest()

    def predict(self, image_bgr: np.ndarray):
        if image_bgr is None:
            raise ValueError("Не удалось декодировать изображение")

        img_hash = self._compute_image_hash(image_bgr)
        if img_hash in self._cache:
            return self._cache[img_hash]

        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224))
        img_normalized = np.float32(img_resized) / 255.0
        
        dummy_confidence = 0.94
        dummy_class = "Pneumonia"
        
        heatmap = cv2.applyColorMap(np.uint8(255 * img_normalized), cv2.COLORMAP_JET)
        _, buffer = cv2.imencode('.jpg', heatmap)
        b64_string = base64.b64encode(buffer).decode('utf-8')

        result = {
            "pathology_class": dummy_class,
            "confidence": dummy_confidence,
            "grad_cam_base64": b64_string
        }

        if img_hash:
            self._cache[img_hash] = result

        return result
