# 🩺 MedecineScan (Variant 25)

![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green)
![Docker](https://img.shields.io/badge/Docker-Enabled-blue)

## 🚀 Стек технологий
* **Бэкенд**: Python, FastAPI, Pydantic V2
* **ИИ**: PyTorch, torchvision, grad-cam, OpenCV
* **Инфраструктура**: Docker, Docker Compose, GitHub Actions с кэшированием

## 🏛 Архитектура сервиса

```mermaid
graph TD
    Client -->|POST predict| API
    API -->|Validation| Pydantic
    Pydantic -->|Tensor| Model
    Model -->|Prediction| GradCAM
    GradCAM -->|Heatmap| API
    API -->|JSON Base64| Client

    Client[Клиент / Врач]
    API[FastAPI Gateway]
    Pydantic[Pydantic V2]
    Model[PyTorch Inference Layer]
    GradCAM[Модуль Grad-CAM XAI]
