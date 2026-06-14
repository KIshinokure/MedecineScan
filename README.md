# 🩺 MedecineScan (Variant 25)
Веб-сервис для классификации медицинских изображений (рентген, КТ) с применением сверточных нейросетей (PyTorch). Сервис возвращает класс патологии и тепловую карту (Grad-CAM), визуализирующую области внимания модели для объяснимости принятых решений.
## 🚀 Стек технологий
* **Бэкенд**: Python, FastAPI, Pydantic V2
* **ИИ**: PyTorch, torchvision, grad-cam, OpenCV
* **Инфраструктура**: Docker, Docker Compose, GitHub Actions
* **Безопасность и тесты**: Pytest, Bandit (SAST), Safety (SCA)
## 🏛 Архитектура сервиса
```mermaid
graph TD
    Client[Клиент / Врач] -->|POST /predict (Файл)| API[FastAPI Gateway]
    API -->|Валидация типа и размера| Pydantic[Pydantic V2]
    Pydantic -->|Тензор изображения| Model[PyTorch Inference Layer]
    Model -->|Предсказание класса патологии| GradCAM[Модуль Grad-CAM XAI]
    GradCAM -->|Наложение тепловой карты| API
    API -->|JSON + Base64 Изображение| Client
```
## ⚙️ Инфраструктура и развертывание
### Запуск через Docker (Рекомендуется)
```bash
docker compose up --build
```
