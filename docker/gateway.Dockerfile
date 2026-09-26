# Gateway: smartrouter + gateway + frontend (served as static files at "/").
# Build from the repo root:  docker build -f docker/gateway.Dockerfile -t llm-gateway .
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY smartrouter/ ./smartrouter/
COPY gateway/ ./gateway/
COPY router_service/ ./router_service/
COPY frontend/ ./frontend/

# Baked in so the gateway can run the router IN-PROCESS if ROUTER_SERVICE_URL is
# left empty. If you deploy router_service as its own Render service instead,
# set ROUTER_SERVICE_URL and this copy is unused (but harmless to keep).
COPY results/router_artifact.joblib ./results/router_artifact.joblib

ENV ROUTER_ARTIFACT_PATH=/app/results/router_artifact.joblib
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s CMD \
  python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "gateway.app:app", "--host", "0.0.0.0", "--port", "8000"]
