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
smartrouter/
  config.py       settings from .env
  clients.py      OllamaClient, OpenAICompatClient (Groq etc.), 429 back-off
  slm_router.py   router prompt + JSON parsing, safe fallback -> HIGH tier
  router.py       SLMRouter: query -> RouteDecision
  labels.py       mixtral_score -> complexity / tier
scripts/
  setup_ubuntu.sh  prepare_dataset.py  verify_phase1.py
  benchmark_slm.py  eval_slm_routing.py  push_to_github.sh
tests/            offline unit tests (labels, parsing)
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

## Next: Phase 2

Train a lightweight classifier on `train.parquet` (predict P(strong needed)), calibrate the
threshold on `val.parquet`, report the cost-vs-quality curve on `test.parquet`, and expose
`POST /route`. The zero-shot numbers from `make eval` are the baseline to beat.
