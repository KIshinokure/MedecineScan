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
    Client["Клиент / Врач"] -->|"POST /predict (Файл)"| API["FastAPI Gateway"]
    API -->|"Валидация"| Pydantic["Pydantic V2"]
    Pydantic -->|"Тензор"| Model["PyTorch Layer"]
    Model -->|"Класс"| GradCAM["Модуль Grad-CAM"]
    GradCAM -->|"Тепловая карта"| API
    API -->|"JSON + Base64"| Client

