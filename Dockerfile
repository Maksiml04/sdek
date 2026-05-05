FROM python:3.11-slim

# Минимальные системные зависимости
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# Убираем --no-cache-dir для лучшего кэширования слоёв
# Добавляем --retries и --timeout для устойчивости
RUN pip install --retries 5 --timeout 30 -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]