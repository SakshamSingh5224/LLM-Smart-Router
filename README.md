# LLM Smart Routing & Cost Optimization System

A cost-optimization gateway that intelligently intercepts user queries, evaluates their complexity using a locally trained classifier, and routes them to either a low-cost or high-cost large language model (LLM) tier. This project is inspired by [RouteLLM](https://github.com/lm-sys/RouteLLM).

**Current Status:** Phase 3 (Gateway, Frontend, and Streaming) is complete.

## 🚀 Features

* **Intelligent Routing:** Uses a custom Logistic Regression classifier trained on TF-IDF features and handcrafted domain signals (e.g., code presence, LaTeX math, reasoning keywords) to predict if a query requires a premium LLM.
* **Streaming UI with LaTeX & Code Formatting:** A plain HTML/JS frontend utilizing Server-Sent Events (SSE) with robust MathJax LaTeX rendering and Markdown code blocks.
* **Auto-Continue Engine:** The backend gateway seamlessly detects API length truncations on the free tier and automatically resumes generation.
* **Cost vs. Quality Control:** Operators can choose routing presets (`aggressive`, `balanced`, `conservative`) or pass explicit thresholds.
* **FastAPI Service:** Exposes a high-performance `POST /route` endpoint with single-digit millisecond latency overhead.

## 🛠️ Free-Tier Technology Stack

| Component | Choice | Cost |
|---|---|---|
| **Router Service** | FastAPI (Python) + Scikit-Learn | Free (Local) |
| **Low-cost Tier** | Local model (e.g., `qwen2.5:1.5b` via Ollama) | Free (Local compute) |
| **High-cost Tier** | `openai/gpt-oss-120b` via Groq | Free (Rate-limited) |
| **Dataset** | [`routellm/gpt4_dataset`](https://huggingface.co/datasets/routellm/gpt4_dataset) | Free |

## 📦 Installation & Setup

1. **Clone the repository and enter the directory:**
   ```bash
   git clone <your-repo-url> llm-smart-router
   cd llm-smart-router
   ```

2. **Set up the virtual environment and install dependencies:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables:**
   Rename `.env.example` to `.env` (or create a new `.env` file) and add your Groq API key:
   ```env
   HIGH_API_KEY=gsk_your_api_key_here
   ```

## 🧠 Phase 2 — Router Classifier

Train the model and generate the `router_artifact.joblib`.
```bash
make data            # Download and split the dataset
make train-router    # Fits logistic regression & calibrates presets
make eval-router     # Generates cost-vs-quality curve on test split
```

## 🌐 Phase 3 — Gateway & Frontend

Wires up the full path: **browser → gateway → router → selected tier → streamed back**.

```bash
# Option A - single service (simplest: router runs in-process inside the gateway)
make serve-gateway                # -> http://localhost:8000 (UI served here)

# Option B - router as its own scalable service
make serve-router                 # terminal 1 -> http://localhost:8001
# (set ROUTER_SERVICE_URL=http://localhost:8001 in .env)
make serve-gateway                # terminal 2 -> http://localhost:8000
```

Open **http://localhost:8000** in your browser. Type a prompt, watch it stream in, see the MathJax formatting apply, and observe which tier answered (LOW/HIGH badge).

## 📡 API Usage (`POST /route`)

**Check Health Status:**
```bash
curl -s http://localhost:8000/health | python3 -m json.tool
```

**Route a Simple Query (Defaults to Low Tier):**
```bash
curl -s -X POST http://localhost:8000/api/chat/stream \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the capital of France?", "explain": true}'
```

**Route a Complex Query (Triggers High Tier):**
*(Note: Use double-backslashes `\\` to properly escape JSON characters like `\sum`)*
```bash
curl -s -X POST http://localhost:8000/api/chat/stream \
     -H "Content-Type: application/json" \
     -d '{"query": "Provide a rigorous mathematical proof using LaTeX \\sum for the time complexity of merging overlapping intervals.", "explain": true}'
```

## 📂 Project Structure

```text
llm-smart-router/
├── frontend/             # Phase 3: HTML/CSS/JS streaming UI (No build step)
├── gateway/              # Phase 3: FastAPI gateway, SSE streaming, caching, auto-continue
├── smartrouter/          # Phase 1/2: Core routing logic, feature extractors, API routes
├── scripts/              # Executable scripts for dataset prep, training, and pushing
├── tests/                # Pytest suite for curves, features, labels, and pipelines
├── data/                 # Local storage for parquet dataset splits
├── results/              # Output directory for the trained .joblib artifact and logs
├── Makefile              # Command orchestration
└── requirements.txt      # Python dependencies
```

## ☁️ Deployment

To push updates to your GitHub repository:
```bash
# Data, results, and .env files are automatically .gitignored to protect secrets.
git add .
git commit -m "Your message"
git push origin main
```

## 🚀 Next: Phase 4
Containerize (`router_service/`, `gateway/`, `frontend/` into Docker images), add Prometheus/Grafana over `results/gateway_log.jsonl`, load-test with k6/Locust, and wire up CI/CD for `router_artifact.joblib` updates.
