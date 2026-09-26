# Router service: smartrouter (shared lib) + router_service + a trained artifact.
# Build from the repo root:  docker build -f docker/router_service.Dockerfile -t llm-router-service .
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1

# System deps for scikit-learn/scipy wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Router service only needs the ML + web layer, not the dataset/embedding extras -
# installing the full requirements.txt is simplest and still small (~200MB image).
RUN pip install --no-cache-dir -r requirements.txt

COPY smartrouter/ ./smartrouter/
COPY gateway/settings.py gateway/router_loader.py ./gateway/
RUN touch gateway/__init__.py
COPY router_service/ ./router_service/

# The trained artifact. Either bake it in at build time (simplest for Render) or
# mount/download it at startup - see README "Phase 4 > Docker" for both options.
COPY results/router_artifact.joblib ./results/router_artifact.joblib

ENV ROUTER_ARTIFACT_PATH=/app/results/router_artifact.joblib
EXPOSE 8001
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s CMD \
  python -c "import urllib.request as u; u.urlopen('http://localhost:8001/health', timeout=3)" || exit 1

CMD ["uvicorn", "router_service.app:app", "--host", "0.0.0.0", "--port", "8001"]
