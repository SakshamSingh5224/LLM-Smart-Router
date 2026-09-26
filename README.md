# LLM Smart Routing & Cost Optimization - Phase 1

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

# get a free key (email only): https://console.groq.com/keys
nano .env                           # set HIGH_API_KEY=gsk_...

source .venv/bin/activate
make test                           # offline unit tests
make data                           # download + label the dataset (~300 MB)
make verify                         # Phase 1 exit-criteria check
make bench                          # cold vs warm router latency
make eval                           # zero-shot SLM routing accuracy (200 prompts)
```

### Phase 1 exit criteria (`make verify`)

1. Ollama is running and the router model is installed
2. Router answers under `ROUTER_LATENCY_BUDGET_MS` (warm p50) with valid JSON
3. Low tier reachable
4. High tier reachable through the API
5. Dataset prepared (`data/processed/{train,val,test}.parquet`)

## Push to GitHub

```bash
# Option A - create an EMPTY repo on github.com/new first, then:
bash scripts/push_to_github.sh https://github.com/<user>/<repo>.git
# (password prompt = a Personal Access Token with `repo` scope)

# Option B - GitHub CLI
sudo apt install -y gh && gh auth login
bash scripts/push_to_github.sh
```
`.env` (your key) and `data/`, `results/` are git-ignored.

## Layout

```
smartrouter/            shared library (Phase 1 + 2)
  config.py              settings from .env
  clients.py              OllamaClient, OpenAICompatClient (Groq etc.), 429 back-off
  slm_router.py, router.py   Phase 1 generative-SLM router (prompt -> JSON -> tier)
  labels.py                mixtral_score -> complexity / tier
  features.py, rules.py    hand-crafted signals + the static fallback router
  featurizers.py           TF-IDF / embedding featurizers for the classifier
  training.py               train + calibrate the Phase 2 classifier
  curves.py                cost-vs-quality curve math
  classifier_router.py      ClassifierRouter (runtime) + circuit breaker

router_service/app.py    Phase 2/3: standalone `POST /route` microservice
gateway/                 Phase 3: backend gateway
  app.py                   POST /api/chat, /api/chat/stream (SSE), /health
  settings.py, cache.py, logging_utils.py, router_loader.py, tier_clients.py
frontend/                Phase 3: plain HTML/CSS/JS chat UI (no build step)

scripts/
  setup_ubuntu.sh, prepare_dataset.py, verify_phase1.py      Phase 1
  benchmark_slm.py, eval_slm_routing.py                      Phase 1
  train_router.py, eval_router_test.py                       Phase 2
  verify_phase3.py                                           Phase 3
  push_to_github.sh
tests/                   offline unit tests (labels, parsing, curves, cache, gateway)
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Ollama is not reachable` | `sudo systemctl start ollama` or `ollama serve &` |
| Router latency above budget on CPU | `ROUTER_MODEL=qwen2.5:0.5b`, or raise `ROUTER_LATENCY_BUDGET_MS`. A BERT-style classifier in Phase 2 brings this to ms. |
| `HTTP 401` from the high tier | wrong/missing `HIGH_API_KEY` |
| `HTTP 404`/"model not offered" | model was renamed; pick one from the list `make verify` prints and set `HIGH_MODEL` |
| `HTTP 429` | free-tier rate limit (about 30 requests/min); wait a minute |
| No GPU / low RAM | keep `qwen2.5:1.5b` or `0.5b`; both run on CPU |
| Want a different free high tier | any OpenAI-compatible endpoint: set `HIGH_BASE_URL`, `HIGH_API_KEY`, `HIGH_MODEL` |

## Phase 2 — Router classifier

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
Compare its numbers to Phase 1's `make eval` — that's the zero-shot baseline this is meant
to beat.

## Phase 3 — Gateway & Frontend

Wires up the full path: **browser → gateway → router → selected tier → streamed back**.

```
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
make train-router                 # need results/router_artifact.joblib first (Phase 2)

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

### Verify Phase 3's exit criterion

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
  exactly the contract from the Phase 2 plan.

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
  scale optimization, not a Phase 3 requirement.

## Next: Phase 4

Containerize (`router_service/`, `gateway/`, `frontend/` are already separable into three
images), add Prometheus/Grafana over `results/gateway_log.jsonl`, load-test with k6/Locust
against `/api/chat`, and wire up CI/CD for `results/router_artifact.joblib` updates
independent of gateway/frontend deploys.
