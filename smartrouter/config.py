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
    ollama_host: str
    router_model: str
    router_latency_budget_ms: float
    low_provider: str
    low_model: str
    low_base_url: str
    low_api_key: str
    high_base_url: str
    high_api_key: str
    high_model: str
    hf_dataset: str
    weak_score_threshold: int
    data_dir: Path
    results_dir: Path
    router_artifact_path: Path
    router_min_confidence: float
    default_route_mode: str
    api_host: str
    api_port: int


def load_settings() -> Settings:
    e = os.getenv
    return Settings(
        ollama_host=e("OLLAMA_HOST", "http://localhost:11434"),
        router_model=e("ROUTER_MODEL", "qwen2.5:1.5b"),
        router_latency_budget_ms=float(e("ROUTER_LATENCY_BUDGET_MS", "1000")),
        low_provider=e("LOW_PROVIDER", "ollama").lower(),
        low_model=e("LOW_MODEL", "qwen2.5:1.5b"),
        low_base_url=e("LOW_BASE_URL", ""),
        low_api_key=e("LOW_API_KEY", ""),
        high_base_url=e("HIGH_BASE_URL", "[https://api.groq.com/openai/v1](https://api.groq.com/openai/v1)"),
        high_api_key=e("HIGH_API_KEY", ""),
        high_model=e("HIGH_MODEL", "openai/gpt-oss-120b"),
        hf_dataset=e("HF_DATASET", "routellm/gpt4_dataset"),
        weak_score_threshold=int(e("WEAK_SCORE_THRESHOLD", "4")),
        data_dir=ROOT / "data",
        results_dir=ROOT / "results",
        router_artifact_path=Path(e("ROUTER_ARTIFACT_PATH", str(ROOT / "results" / "router_classifier.joblib"))),
        router_min_confidence=float(e("ROUTER_MIN_CONFIDENCE", "0.0")),
        default_route_mode=e("DEFAULT_ROUTE_MODE", "balanced"),
        api_host=e("API_HOST", "0.0.0.0"),
        api_port=int(e("API_PORT", "8000")),
    ) 
