# Gateway image. Build from repo root.
FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY smartrouter/ ./smartrouter/
COPY gateway/ ./gateway/
COPY router_service/ ./router_service/
COPY frontend/ ./frontend/
COPY results/router_artifact.joblib ./results/router_artifact.joblib
ENV ROUTER_ARTIFACT_PATH=/app/results/router_artifact.joblib
EXPOSE 8000
CMD ["sh", "-c", "uvicorn gateway.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
