# LLM Smart Routing & Cost Optimization - Phase 1A

Environment, free model access, and the routing dataset for a gateway that sends
simple prompts to a cheap model and hard prompts to a strong one (inspired by
[RouteLLM](https://github.com/lm-sys/RouteLLM)).

**Everything here is free.** No credit card, no paid API.

| Component | Choice | Cost |
|---|---|---|
| Router SLM (local) | `qwen2.5:1.5b` via [Ollama](https://ollama.com) | free, runs on CPU or GPU |
| Low-cost tier | same local model (`LOW_PROVIDER=ollama`) | free |
| High-cost tier | `openai/gpt-oss-120b` on [Groq](https://console.groq.com) free tier | free, rate-limited |
| Dataset | [`routellm/gpt4_dataset`](https://huggingface.co/datasets/routellm/gpt4_dataset) (~119k GPT-4-judged prompts) | free |

> **Why not Llama-3.x on Groq?** Groq announced shutdown of `llama-3.1-8b-instant` and
> `llama-3.3-70b-versatile` on 2026-08-16 and recommends `openai/gpt-oss-20b` /
> `openai/gpt-oss-120b`. Model catalogs on free tiers change often, so every model name
> is a setting in `.env`, and `make verify` lists what your key can actually use.

## How the dataset becomes routing labels

Each row of `routellm/gpt4_dataset` has a `prompt` and a `mixtral_score` (1-5): how good
a *weak* model's answer was, as judged by GPT-4.

| mixtral_score | complexity | routing label |
|---|---|---|
| 5 | simple | `low` |
| 4 | medium | `low` |
| 1-3 | complex | `high` |

Change the cut-off with `WEAK_SCORE_THRESHOLD` in `.env`.
Splits: `test` = the dataset's own validation split (held out), `train`/`val` = 90/10
stratified split of its train split. Duplicates and train/test overlap are removed.

## Quick start (Ubuntu 22.04 / 24.04)

```bash
git clone <your-repo-url> llm-smart-router && cd llm-smart-router

bash scripts/setup_ubuntu.sh        # apt deps, venv, Ollama, pulls the model(s)

# get a free key (email only): [https://console.groq.com/keys](https://console.groq.com/keys)
nano .env                           # set HIGH_API_KEY=gsk_...

source .venv/bin/activate
make test                           # offline unit tests
make data                           # download + label the dataset (~300 MB)
make verify                         # Phase 1A exit-criteria check
make bench                          # cold vs warm router latency
make eval                           # zero-shot SLM routing accuracy (200 prompts)

```

### Phase 1A exit criteria (`make verify`)

1. Ollama is running and the router model is installed
2. Router answers under `ROUTER_LATENCY_BUDGET_MS` (warm p50) with valid JSON
3. Low tier reachable
4. High tier reachable through the API
5. Dataset prepared (`data/processed/{train,val,test}.parquet`)

## Push to GitHub

```bash
# Option A - create an EMPTY repo on [https://github.com/new](https://github.com/new) first, then:
bash scripts/push_to_github.sh [https://github.com/](https://github.com/)<user>/<repo>.git
# (password prompt = a Personal Access Token with `repo` scope)

# Option B - GitHub CLI
sudo apt install -y gh && gh auth login
bash scripts/push_to_github.sh

```

`.env` (your key) and `data/`, `results/` are git-ignored.

## Layout

```text
smartrouter/             shared library (Phase 1A + 1B)
  config.py              settings from .env
  clients.py             OllamaClient, OpenAICompatClient (Groq etc.), 429 back-off
  slm_router.py, router.py   Phase 1A generative-SLM router (prompt -> JSON -> tier)
  labels.py              mixtral_score -> complexity / tier
  features.py, rules.py  hand-crafted signals + the static fallback router
  featurizers.py         TF-IDF / embedding featurizers for the classifier
  training.py            train + calibrate the Phase 1B classifier
  curves.py              cost-vs-quality curve math
  classifier_router.py   ClassifierRouter (runtime) + circuit breaker

router_service/app.py    Phase 1B/1C: standalone `POST /route` microservice
gateway/                 Phase 1C: backend gateway
  app.py                 POST /api/chat, /api/chat/stream (SSE), /health
  settings.py, cache.py, logging_utils.py, router_loader.py, tier_clients.py
frontend/                Phase 1C: plain HTML/CSS/JS chat UI (no build step)

scripts/
  setup_ubuntu.sh, prepare_dataset.py, verify_phase1.py      Phase 1A
  benchmark_slm.py, eval_slm_routing.py                      Phase 1A
  train_router.py, eval_router_test.py                       Phase 1B
  verify_phase3.py                                           Phase 1C
  push_to_github.sh
tests/                   offline unit tests (labels, parsing, curves, cache, gateway)

```

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `Ollama is not reachable` | `sudo systemctl start ollama` or `ollama serve &` |
| Router latency above budget on CPU | `ROUTER_MODEL=qwen2.5:0.5b`, or raise `ROUTER_LATENCY_BUDGET_MS`. A BERT-style classifier in Phase 1B brings this to ms. |
| `HTTP 401` from the high tier | wrong/missing `HIGH_API_KEY` |
| `HTTP 404`/"model not offered" | model was renamed; pick one from the list `make verify` prints and set `HIGH_MODEL` |
| `HTTP 429` | free-tier rate limit (about 30 requests/min); wait a minute |
| No GPU / low RAM | keep `qwen2.5:1.5b` or `0.5b`; both run on CPU |
| Want a different free high tier | any OpenAI-compatible endpoint: set `HIGH_BASE_URL`, `HIGH_API_KEY`, `HIGH_MODEL` |

## Phase 1B — Router classifier

```bash
make train-router     # trains + calibrates -> results/router_artifact.joblib
make eval-router       # cost-vs-quality curve on the held-out TEST split

```

`scripts/train_router.py` fits a logistic-regression classifier on TF-IDF (+ hand-crafted
features; add `--backends tfidf embed` to also try `BAAI/bge-small-en-v1.5` sentence
embeddings via `fastembed`, no PyTorch required) on `train.parquet`, picks the best
backend/`C` by **validation** ROC-AUC, then calibrates three thresholds on `val.parquet`
("aggressive"/"balanced"/"conservative" cost-quality presets — how aggressively to prefer
the cheap tier). `scripts/eval_router_test.py` then scores the untouched `test.parquet`
split and reports the cost-vs-quality curve (`results/test_cost_quality_curve.{csv,png}`).
Compare its numbers to Phase 1A's `make eval` — that's the zero-shot baseline this is meant
to beat.

## Phase 1C — Gateway & Frontend

Wires up the full path: **browser → gateway → router → selected tier → streamed back**.

```text
frontend/ (plain HTML/CSS/JS, no build step)
        │  fetch() + SSE
        ▼
gateway/app.py  (FastAPI, :8000)
        │  in-process, OR HTTP if ROUTER_SERVICE_URL is set
        ▼
router_service/app.py  (FastAPI, :8001, optional standalone deploy)
        │  ClassifierRouter loads results/router_artifact.joblib
        │  falls back to smartrouter.rules if missing/broken (circuit breaker)
        ▼
gateway/tier_clients.py → Ollama (low) or Groq (high), streamed

```

### Run it

```bash
make train-router                 # need results/router_artifact.joblib first (Phase 1B)

# Option A - single service (simplest: router runs in-process inside the gateway)
make serve-gateway                # -> http://localhost:8000  (frontend is served here too)

# Option B - router as its own scalable service (matches the plan's architecture)
make serve-router                 # terminal 1 -> http://localhost:8001
# then in .env set ROUTER_SERVICE_URL=http://localhost:8001
make serve-gateway                # terminal 2 -> http://localhost:8000

```

Open **http://localhost:8000** — type a prompt, watch it stream in, see which tier
answered (LOW/HIGH badge), the routing probability, and (if you set `COST_PER_1K_*`)
estimated cost. Tick "explain routing" to see the classifier's reasoning per request.

### Verify Phase 1C's exit criterion

> *"End-to-end flow works — a user submits a query, it's routed, and a streamed response
> renders in the UI."*

```bash
make verify3     # gateway must already be running (make serve-gateway, separate terminal)

```

Checks: `/health`, a full non-streaming `/api/chat` round trip, that
`/api/chat/stream` emits `meta` → `delta`(s) → `done` in order (proving the whole
pipeline runs end-to-end), that forcing `low`/`high` both work, and that a repeated
query is served from cache.

### API

* `POST /api/chat` — `{query, mode?, threshold?, force_tier?, use_cache?, explain?}` → full
JSON answer + the routing decision + latency/cost breakdown.
* `POST /api/chat/stream` — same body, Server-Sent Events: `meta` (routing decision) →
`delta` (token chunks) → `done` (totals). `force_tier` is a debug override for A/B
testing "always cheap" vs "always strong" vs "routed", per the plan's testing section.
* `GET /health`, `GET /modes`, `GET /logs/recent?n=20`
* Router service (if run standalone): `POST /route`, `GET /health`, `GET /modes` —
exactly the contract from the Phase 1B plan.

### What's intentionally simple (and the free/no-infra reason why)

* **Cache** is in-process (normalized-text match + LRU/TTL), not Redis + embedding
similarity. No server to install, and it already skips repeated generation calls;
`gateway/cache.py` documents the swap-in point if you add Redis later.
* **Frontend** is plain HTML/CSS/JS — no Node/npm/build step, one less thing to install
on a fresh Ubuntu box. `fetch()` + manual SSE parsing stands in for `EventSource`
because the stream needs a POST body.
* **Cost tracking** defaults to $0 (`COST_PER_1K_LOW/HIGH=0.0`) since both tiers are free;
set them to simulate what a paid deployment's cost dashboard would show.
* Gateway ↔ router is plain REST (`httpx`), not gRPC — the plan lists gRPC as a
scale optimization, not a Phase 1C requirement.

## Phase 2A — Identity & Database Foundation (Completed)

MVP 2 adds identity and budget control on top of the routing gateway.

* **Database Integration:** SQLAlchemy 2.0 ORM managing `users`, `refresh_tokens`, `policies`, `user_policy`, and `usage_ledger` tables via Postgres (Neon) or SQLite.
* **Authentication:** Secure user registration (`/api/auth/register`) and login (`/api/auth/login`) using bcrypt password hashing and JWT access/refresh tokens.
* **Protected Endpoints:** Core API endpoints (`/api/chat`, `/api/chat/stream`) now require a valid Bearer JWT header.
* **Frontend UI:** Dark-mode login and registration interface integrated seamlessly with the chat application, managing local session state and handling 401 redirects.

## Phase 2B — Policy Engine & Usage Enforcement (Completed)

* Implementing per-user quotas, auto-downgrading when budget is exhausted, and forcing tier overrides independent of the classifier.

## Phase 2C — Admin Dashboard & Observability (Completed)

* Building UI for managing user policies and integrating Grafana/Prometheus metrics for policy decisions (`policy_decision_total`).

## Phase 3 — Intelligent Routing with Local Knowledge Base (MVP-3)

**1. Problem Statement**
Current LLMs are costly and hallucinate on proprietary / Indian-context data. For every query, we call a paid external LLM even if the answer already exists in our own documents like ISRO reports, NCERT books, or SC judgments.

**2. Objective of MVP-3**
To build a **Local RAG Agent** that:

1. Intercepts queries and intelligently routes proprietary/Indian-context queries to a local Vector DB.
2. Answers from local PDFs/transcripts without calling OpenAI.
3. Proves cost saving via semantic cache and metering.

**3. Scope of MVP-3**

* **In Scope:** Vector DB with 80 Indian-context PDFs, Local-KB-Agent (Retrieval + Re-ranking), Routing Intelligence [Judge Model], Semantic Cache (Redis), Demo UI + Cost Saving Dashboard.
* **Out of Scope:** Real-time document upload, User Auth, Multi-language Hindi support [for MVP-4].

**4. System Architecture - Routing Intelligence**

```text
User Query
 |
 v
[Judge Model - Intent Classifier]
 |--- Intent = proprietary/indian_context (score > 0.7) ---> [Semantic Cache Check]
 | |-> Hit? Return Answer (Cost = 0)
 | |-> Miss? -> [Vector DB Search (Qdrant/pgvector)] -> [Re-ranker] -> [Small Local LLM - Phi-3] -> Answer + Save to Cache
 |
 |--- Intent = general_world_knowledge ---> [Existing Router - Phase 1/2] -> External LLM

```

**5. Requirement Specification**

* **Dataset (Bharat Knowledge Base):** ISRO Archive (30 PDFs), NCERT Class 12 (5 Books), SC Judgments (20 cases).
* **Functional Requirements:**
* **FR1 - Ingestion:** Extract, chunk (600 tokens/100 overlap), embed using `bge-small-en-v1.5`.
* **FR2 - Judge Intelligence:** Label queries (`local_kb` or `external_llm`) based on keywords (ISRO, NCERT, SC, Budget, Chandrayaan).
* **FR3 - Retrieval:** Retrieve Top 10, re-rank to Top 3 (`bge-reranker-base`).
* **FR4 - Threshold:** If similarity < 0.78, fallback to external LLM. If > 0.78, answer from context.
* **FR5 - Semantic Cache:** Redis caching (`query_embedding -> answer`, similarity > 0.92).
* **FR6 - Dashboard:** Show logs for routing and cost savings.



**6. Tech Stack for MVP-3**

* **Vector DB:** Qdrant [Local Docker] or pgvector
* **Models:** BAAI/bge-small-en-v1.5 (Embedding), BAAI/bge-reranker-base (Re-ranker), Phi-3 Mini / Mistral 7B Q4 via Ollama (Local LLM)
* **Infrastructure:** Redis (Cache), Python + FastAPI (Backend), Streamlit (Frontend Demo)

**7. Success Metrics**

1. 40%+ queries served from Local DB.
2. 30%+ queries served from Semantic Cache on repeat.
3. Cost per 100 queries reduced by 50% vs MVP-2.
4. Hallucination rate for Indian-context queries < 5%.




## MVP 3C — Qdrant Retrieval + Re-ranking

MVP 3C uses a two-stage local retrieval pipeline:

```text
Query
  -> BAAI/bge-small-en-v1.5
  -> Qdrant Top-10
  -> BAAI/bge-reranker-base
  -> Top-3 context
  -> relevance gate (default 0.78)
```

Source metadata is retained with every selected chunk. Reranker latency and raw
CrossEncoder scores are recorded separately from normalized 0–1 presentation
scores. The normalized sigmoid value is **not a calibrated probability**.

Run local validation after Qdrant is populated:

```bash
source .venv/bin/activate
PYTHONPATH=. python -m scripts.test_mvp3c "What was the objective of Chandrayaan-3?"
PYTHONPATH=. python -m scripts.calibrate_mvp3c
pytest -q tests/test_mvp3_retriever.py
```

Calibration intentionally does not change the required 0.78 threshold automatically.
It reports positive/negative score separation so the threshold can be defended or
revised based on measured knowledge-base behavior.

The exact `bge-reranker-base` pipeline is intended for local development. The
Render Free deployment disables MVP 3C retrieval by default because the full
reranker is not appropriate for a 512 MB service; Qdrant can remain external.
