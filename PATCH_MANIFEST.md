# Phase 3F patch package

This package contains every new or changed file relative to the uploaded `LLM-Smart-Router-main.zip`, including the already-supplied frontend/gateway patch content. It does not include `.env`, access tokens, API keys, model caches, or local database files.

## Apply to an existing checkout

From the repository root, extract this ZIP somewhere temporary, then copy the contents of `LLM-Smart-Router-Phase3F-Patch/` into the repository root, preserving paths. Review your own `.env` before rebuilding. Do not run `docker compose down -v` if you need to preserve Qdrant/Redis data.

## Files included

- `.env.example`
- `Makefile`
- `PHASE3F_CHANGELOG.md`
- `README.md`
- `docker/docker-compose.yml`
- `docker/grafana/dashboards/llm-smart-router-phase3f.json`
- `docker/grafana/provisioning/dashboards/provider.yml`
- `docker/grafana/provisioning/datasources/prometheus.yml`
- `docker/router-requirements.txt`
- `docker/router_service.Dockerfile`
- `frontend/app.js`
- `frontend/chat.html`
- `frontend/config.js`
- `frontend/index.html`
- `frontend/style.css`
- `gateway/app.py`
- `gateway/cache.py`
- `gateway/db/database.py`
- `gateway/metrics.py`
- `gateway/policy.py`
- `gateway/reranker.py`
- `gateway/retriever.py`
- `render.yaml`
- `scripts/verify_phase3f.py`
- `tests/test_mvp3d_gateway.py`
- `tests/test_mvp3e_cache.py`
- `tests/test_mvp3f_metrics.py`
