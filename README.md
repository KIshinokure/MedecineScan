# MedecineScan (Variant 25)

Веб-сервис для классификации медицинских изображений (рентген, КТ) с применением сверточных нейросетей (PyTorch).

## Архитектура системы

```mermaid
graph TD
    Client -->|POST /predict| API
    API -->|Validation| Pydantic
    Pydantic -->|Tensor| Model
    Model -->|Prediction| GradCAM
    GradCAM -->|Heatmap| API
    API -->|JSON + Base64| Client

    Client[Клиент / Врач]
    API[FastAPI Gateway]
    Pydantic[Pydantic V2]
    Model[PyTorch Inference Layer]
    GradCAM[Модуль Grad-CAM XAI]
Быстрый старт через Docker
Bash
docker compose up --build
