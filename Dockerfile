FROM python:3.11-slim

# Явно задаем рабочую директорию в самом начале
WORKDIR /app

# Устанавливаем системные зависимости, необходимые для OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Копируем зависимости и устанавливаем их
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем исходный код приложения
COPY ./app ./app

# Открываем порт и прописываем команду для запуска (например, uvicorn)
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]