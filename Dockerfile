FROM python:3.11-slim

# Минимальные системные зависимости
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копируем требования и устанавливаем зависимости
COPY requirements.txt .
RUN pip install --retries 5 --timeout 30 -r requirements.txt

# Копируем весь код проекта
COPY . .

# Открываем порт внутри контейнера
EXPOSE 8000

# Запускаем приложение
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]