# LLM Smart Routing & Cost Optimization System

A cost-optimization gateway that intelligently intercepts user queries, evaluates their complexity using a locally trained classifier, and routes them to either a low-cost or high-cost large language model (LLM) tier. This project is inspired by [RouteLLM](https://github.com/lm-sys/RouteLLM).

**Current Status:** Phase 1 (Dataset & Baseline) and Phase 2 (Trained Classifier & FastAPI Router Service) are complete.

## 🚀 Features

* **Intelligent Routing:** Uses a custom Logistic Regression classifier trained on TF-IDF features and handcrafted domain signals (e.g., code presence, LaTeX math, reasoning keywords) to predict if a query requires a premium LLM.
* **Cost vs. Quality Control:** Operators can choose routing presets (`aggressive`, `balanced`, `conservative`) or pass explicit thresholds to dynamically balance API costs against response quality.
* **FastAPI Service:** Exposes a high-performance `POST /route` endpoint with single-digit millisecond latency overhead.
* **Safe Fallbacks:** If the classifier artifact is missing or fails to parse a query, the system gracefully degrades to a deterministic, rule-based heuristic router without dropping the user's request.

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

## 🧠 Phase 2 Execution (Training & Serving)

The project includes a `Makefile` to simplify all critical operations. Run these sequentially from within your activated virtual environment:

```bash
# 1. Run offline unit tests to ensure environment stability
make test

# 2. Download and label the dataset (creates train, val, and test splits)
make data

# 3. Train the classifier and calibrate cost/quality thresholds
make train

# 4. Evaluate the trained artifact against the validation set
make eval-classifier

# 5. Start the FastAPI Router Service on localhost:8000
make serve
```

## 📡 API Usage (`POST /route`)

While `make serve` is running, you can interact with the router via REST API.

**Check Health Status:**
```bash
curl -s http://localhost:8000/health | python3 -m json.tool
```

**Route a Simple Query (Defaults to Low Tier):**
```bash
curl -s -X POST http://localhost:8000/route \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the capital of France?", "explain": true}'
```

**Route a Complex Query (Triggers High Tier):**
*Note: Use double-backslashes (`\\`) to properly escape JSON characters like `\sum`.*
```bash
curl -s -X POST http://localhost:8000/route \
     -H "Content-Type: application/json" \
     -d '{"query": "Provide a rigorous mathematical proof using LaTeX \\sum for the time complexity of merging overlapping intervals.", "explain": true}'
```

**Override Thresholds Manually:**
```bash
curl -s -X POST http://localhost:8000/route \
     -H "Content-Type: application/json" \
     -d '{"query": "Write a Python function.", "threshold": 0.04, "explain": true}'
```

## 📂 Project Structure

```text
llm-smart-router/
├── smartrouter/          # Core routing logic, feature extractors, and API routes
│   ├── api.py            # FastAPI application and endpoints
│   ├── classifier_router.py # Trained ML router
│   ├── rules.py          # Fallback heuristic router
│   └── training.py       # Model training and calibration pipeline
├── scripts/              # Executable scripts for dataset prep, training, and pushing
├── tests/                # Pytest suite for curves, features, labels, and pipelines
├── data/processed/       # Local storage for parquet dataset splits
├── results/              # Output directory for the trained .joblib artifact and metrics
├── Makefile              # Command orchestration
└── requirements.txt      # Python dependencies
```

## ☁️ Deployment

To push this project to a new GitHub repository, utilize the provided deployment script:
```bash
bash scripts/push_to_github.sh [https://github.com/](https://github.com/)<your-username>/<your-repo-name>.git
```
*(Note: Data, results, and `.env` files are automatically `.gitignore`d to protect secrets and save space).*
