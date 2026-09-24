"""Central settings loaded from environment variables or .env file."""
from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

@dataclass(frozen=True)
class Settings:
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    router_model: str = os.getenv("ROUTER_MODEL", "qwen2.5:1.5b")
    router_latency_budget_ms: float = float(os.getenv("ROUTER_LATENCY_BUDGET_MS", "1000"))
    low_provider: str = os.getenv("LOW_PROVIDER", "ollama").lower()
    low_model: str = os.getenv("LOW_MODEL", "qwen2.5:1.5b")
    low_base_url: str = os.getenv("LOW_BASE_URL", "")
    low_api_key: str = os.getenv("LOW_API_KEY", "")
    high_base_url: str = os.getenv("HIGH_BASE_URL", "[https://api.groq.com/openai/v1](https://api.groq.com/openai/v1)")
    high_api_key: str = os.getenv("HIGH_API_KEY", "")
    high_model: str = os.getenv("HIGH_MODEL", "openai/gpt-oss-120b")
    hf_dataset: str = os.getenv("HF_DATASET", "routellm/gpt4_dataset")
    weak_score_threshold: int = int(os.getenv("WEAK_SCORE_THRESHOLD", "4"))
    data_dir: Path = ROOT / "data"
    results_dir: Path = ROOT / "results"
    router_artifact_path: Path = Path(os.getenv("ROUTER_ARTIFACT_PATH", str(ROOT / "results" / "router_classifier.joblib")))
    router_min_confidence: float = float(os.getenv("ROUTER_MIN_CONFIDENCE", "0.0"))
    default_route_mode: str = os.getenv("DEFAULT_ROUTE_MODE", "balanced")
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))

def load_settings() -> Settings:
    return Settings()
