#!/usr/bin/env bash
# One-shot Ubuntu (22.04 / 24.04) setup for Phase 1.
# Usage:  bash scripts/setup_ubuntu.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> [1/5] System packages"
sudo apt-get update -y
sudo apt-get install -y git curl zstd python3 python3-venv python3-pip build-essential

echo "==> [2/5] Python virtualenv + dependencies"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
[ -f .env ] || { cp .env.example .env; echo "    created .env (edit it to add your Groq key)"; }

echo "==> [3/5] Ollama (local LLM server)"
if ! command -v ollama >/dev/null 2>&1; then
  curl -fsSL https://ollama.com/install.sh | sh
else
  echo "    ollama already installed: $(ollama --version 2>&1 | head -1)"
fi

echo "==> [4/5] Make sure the Ollama server is running"
if ! curl -fs http://localhost:11434/api/tags >/dev/null 2>&1; then
  sudo systemctl enable --now ollama 2>/dev/null || true
  for _ in $(seq 1 20); do curl -fs http://localhost:11434/api/tags >/dev/null 2>&1 && break || sleep 1; done
fi
if ! curl -fs http://localhost:11434/api/tags >/dev/null 2>&1; then
  echo "    systemd service unavailable - starting 'ollama serve' in the background"
  nohup ollama serve > /tmp/ollama.log 2>&1 &
  for _ in $(seq 1 20); do curl -fs http://localhost:11434/api/tags >/dev/null 2>&1 && break || sleep 1; done
fi
curl -fs http://localhost:11434/api/tags >/dev/null || { echo "Ollama did not start. See /tmp/ollama.log"; exit 1; }

echo "==> [5/5] Pull local models (from .env)"
ROUTER_MODEL=$(grep -E '^ROUTER_MODEL=' .env | cut -d= -f2- || true); ROUTER_MODEL=${ROUTER_MODEL:-qwen2.5:1.5b}
LOW_MODEL=$(grep -E '^LOW_MODEL=' .env | cut -d= -f2- || true);       LOW_MODEL=${LOW_MODEL:-qwen2.5:1.5b}
LOW_PROVIDER=$(grep -E '^LOW_PROVIDER=' .env | cut -d= -f2- || true);  LOW_PROVIDER=${LOW_PROVIDER:-ollama}
ollama pull "$ROUTER_MODEL"
if [ "$LOW_PROVIDER" = "ollama" ] && [ "$LOW_MODEL" != "$ROUTER_MODEL" ]; then ollama pull "$LOW_MODEL"; fi

cat <<MSG

Setup complete. Next:
  1. Get a FREE Groq key (no card): https://console.groq.com/keys
     then put it in .env as HIGH_API_KEY=...
  2. source .venv/bin/activate
  3. make data      # download + label the RouteLLM dataset
  4. make verify    # Phase 1 exit-criteria check
  5. make bench && make eval   # Phase 1 latency + zero-shot baseline
  6. make train && make eval-classifier && make serve   # Phase 2: trained classifier + POST /route
MSG
