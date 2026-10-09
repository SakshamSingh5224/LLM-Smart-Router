# Phase 3F implementation summary

## Implemented

- Centralized Prometheus metrics for local routes, semantic/exact cache hits, retrieval/reranking latency, reranker model-load latency, local generation, local TTFT, external fallback reasons, estimated cost avoided, and usage-ledger write failures.
- Added a provisioned Grafana dashboard and explicit Prometheus datasource UID.
- Added `/api/system/status` Phase 3F observability metadata and UI indicators for TTFT, estimated avoided cost, and RAG source documents.
- Persisted retrieved source metadata with exact/semantic cache entries and returned it to the UI on cache hits.
- Moved blocking retrieval/reranking into a worker thread and guarded lazy model initialization/inference for predictable concurrency.
- Added external-router fallback when retrieval/reranking raises or context is insufficient; local-generation failure fallback remains instrumented.
- Ended the policy read transaction before expensive model work, enabled Postgres pool pre-ping/recycling, and made usage writes best-effort with rollback and a failure metric.
- Recorded cached requests in the usage ledger with zero generated tokens/cost so cache hits still count against query quotas.
- Added a smoke-check script and Phase 3F regression tests.
- Reintroduced Qdrant into the primary Compose stack and persisted Qdrant, Redis, and model caches in named volumes. Router image uses a slim requirements file to avoid unrelated heavy ingestion packages.

## Validation performed in the packaging environment

- 10 selected unit tests passed: policy, semantic cache, source metadata persistence, and usage-ledger failure handling.
- Python compile checks passed for project modules.
- Frontend JavaScript syntax check passed.
- Compose/Prometheus/Grafana provisioning YAML and Grafana dashboard JSON parsed successfully.
- Prometheus custom metric families were checked for presence.

The packaging environment does not include Docker or the full gateway runtime/model dependencies, so the complete FastAPI end-to-end suite and the live local-stack smoke check still need to run on the target Ubuntu machine. Run `python scripts/verify_phase3f.py --base-url http://127.0.0.1:8000` there; set `GATEWAY_TEST_TOKEN` to additionally test authenticated chat and usage reporting.

`local_rag_accuracy_ratio` is intentionally reserved for Phase 3G evaluation; serving traffic alone is not a valid accuracy measurement.
