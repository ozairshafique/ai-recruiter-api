FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl gcc g++ libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/data/temp /app/logs /app/faiss_index

RUN useradd -m appuser && chown -R appuser:appuser /app
RUN chmod -R 755 /app/data/temp /app/logs /app/faiss_index
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]