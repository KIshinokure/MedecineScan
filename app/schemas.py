from pydantic import BaseModel, Field

class HealthCheck(BaseModel):
    status: str
    model_loaded: bool

class PredictionResponse(BaseModel):
    pathology_class: str = Field(..., description="Класс патологии (например, Пневмония)")
    confidence: float = Field(..., description="Уверенность модели от 0 до 1")
    grad_cam_base64: str = Field(..., description="Тепловая карта в формате Base64")
