"""Gateway settings: model config plus gateway-only options."""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path
from smartrouter.config import Settings as ModelSettings
from smartrouter.config import load_settings as load_model_settings

ROOT = Path(__file__).resolve().parents[1]

@dataclass(frozen=True)
class GatewaySettings:
    model: ModelSettings
    host: str
    port: int
    router_mode: str
    router_artifact_path: Path
    router_service_url: str
    router_timeout_s: float
    max_tokens: int
    cache_enabled: bool
    cache_size: int
    cache_ttl_s: float
    cors_origins: list = field(default_factory=list)
    log_path: Path = ROOT / "results" / "gateway_log.jsonl"
    cost_per_1k_low: float = 0.0
    cost_per_1k_high: float = 0.0
    api_key: str = ""
    rate_limit_per_min: int = 30

    enable_mvp3_routing: bool = True
    intent_threshold: float = 0.70
    indian_context_keywords: list = field(default_factory=lambda: [
        "isro", "ncert", "chandrayaan", "supreme court",
        "indian budget", "indian regulation", "bharat",
    ])

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "bharat_knowledge_base"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    retrieval_top_k: int = 10
    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_candidate_k: int = 10
    rerank_top_k: int = 3
    reranker_max_length: int = 512
    rerank_relevance_threshold: float = 0.70
    enable_mvp3_retrieval: bool = True
    enable_mvp3d_rag: bool = True
    local_rag_model: str = "qwen2.5:1.5b"
    local_rag_temperature: float = 0.2

    def __post_init__(self):
        override = os.getenv("GATEWAY_LOG_PATH")
        if override:
            object.__setattr__(self, "log_path", Path(override))
        if self.router_service_url and not self.router_service_url.startswith(("http://", "https://")):
            object.__setattr__(self, "router_service_url", f"http://{self.router_service_url}")

def load_gateway_settings() -> GatewaySettings:
    e = os.getenv
    origins = e("GATEWAY_CORS_ORIGINS", "*")
    raw_keywords = e("INDIAN_CONTEXT_KEYWORDS", "")
    keywords = [x.strip() for x in raw_keywords.split(",") if x.strip()] if raw_keywords else [
        "isro", "ncert", "chandrayaan", "supreme court",
        "indian budget", "indian regulation", "bharat",
    ]
    return GatewaySettings(
        model=load_model_settings(),
        host=e("GATEWAY_HOST", "0.0.0.0"),
        port=int(e("PORT", e("GATEWAY_PORT", "8000"))),
        router_mode=e("ROUTER_MODE", "balanced"),
        router_artifact_path=Path(e("ROUTER_ARTIFACT_PATH", str(ROOT / "results" / "router_artifact.joblib"))),
        router_service_url=e("ROUTER_SERVICE_URL", "").rstrip("/"),
        router_timeout_s=float(e("ROUTER_TIMEOUT_S", "5.0")),
        max_tokens=int(e("GATEWAY_MAX_TOKENS", "512")),
        cache_enabled=e("GATEWAY_CACHE_ENABLED", "true").lower() in ("1", "true", "yes"),
        cache_size=int(e("GATEWAY_CACHE_SIZE", "500")),
        cache_ttl_s=float(e("GATEWAY_CACHE_TTL_S", "3600")),
        cors_origins=["*"] if origins.strip() == "*" else [x.strip() for x in origins.split(",") if x.strip()],
        cost_per_1k_low=float(e("COST_PER_1K_LOW", "0.0")),
        cost_per_1k_high=float(e("COST_PER_1K_HIGH", "0.0")),
        api_key=e("GATEWAY_API_KEY", ""),
        rate_limit_per_min=int(e("GATEWAY_RATE_LIMIT_PER_MIN", "30")),
        enable_mvp3_routing=e("ENABLE_MVP3_ROUTING", "true").lower() in ("1", "true", "yes"),
        intent_threshold=float(e("INTENT_THRESHOLD", "0.70")),
        indian_context_keywords=keywords,
        qdrant_url=e("QDRANT_URL", "http://localhost:6333").rstrip("/"),
        qdrant_api_key=e("QDRANT_API_KEY", ""),
        qdrant_collection=e("QDRANT_COLLECTION", "bharat_knowledge_base"),
        embedding_model=e("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        retrieval_top_k=int(e("RETRIEVAL_TOP_K", "10")),
        reranker_model=e("RERANKER_MODEL", "BAAI/bge-reranker-base"),
        reranker_candidate_k=int(e("RERANKER_CANDIDATE_K", "10")),
        rerank_top_k=int(e("RERANK_TOP_K", "3")),
        reranker_max_length=int(e("RERANKER_MAX_LENGTH", "512")),
        rerank_relevance_threshold=float(e("RERANK_RELEVANCE_THRESHOLD", "0.70")),
        enable_mvp3_retrieval=e("ENABLE_MVP3_RETRIEVAL", "true").lower() in ("1", "true", "yes"),
        enable_mvp3d_rag=e("ENABLE_MVP3D_RAG", "true").lower() in ("1", "true", "yes"),
        local_rag_model=e("LOCAL_RAG_MODEL", e("LOW_MODEL", "qwen2.5:1.5b")),
        local_rag_temperature=float(e("LOCAL_RAG_TEMPERATURE", "0.2")),
    )
