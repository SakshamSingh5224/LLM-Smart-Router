"""Gateway settings: model config (from smartrouter.config) + gateway-only options."""
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
    router_mode: str                    # default cost-quality preset: aggressive|balanced|conservative
    router_artifact_path: Path
    router_service_url: str             # empty => call the router in-process (no second HTTP hop)
    router_timeout_s: float
    max_tokens: int
    cache_enabled: bool
    cache_size: int
    cache_ttl_s: float
    cors_origins: list = field(default_factory=list)
    log_path: Path = ROOT / "results" / "gateway_log.jsonl"
    cost_per_1k_low: float = 0.0        # both free tiers by default -> $0
    cost_per_1k_high: float = 0.0
    api_key: str = ""                   # if set, /api/* requires header X-API-Key: <this>
    rate_limit_per_min: int = 30        # per API key (or per IP if no key configured)

    def __post_init__(self):
        override = os.getenv("GATEWAY_LOG_PATH")
        if override:
            object.__setattr__(self, "log_path", Path(override))
        url = self.router_service_url
        if url and not url.startswith(("http://", "https://")):
            # Some hosts (e.g. Render's `hostport` service reference) return a
            # bare host:port with no scheme - assume plain HTTP on that case.
            object.__setattr__(self, "router_service_url", f"http://{url}")


def load_gateway_settings() -> GatewaySettings:
    e = os.getenv
    origins = e("GATEWAY_CORS_ORIGINS", "*")
    return GatewaySettings(
        model=load_model_settings(),
        host=e("GATEWAY_HOST", "0.0.0.0"),
        port=int(e("GATEWAY_PORT", "8000")),
        router_mode=e("ROUTER_MODE", "balanced"),
        router_artifact_path=Path(e("ROUTER_ARTIFACT_PATH", str(ROOT / "results" / "router_artifact.joblib"))),
        router_service_url=e("ROUTER_SERVICE_URL", "").rstrip("/"),
        router_timeout_s=float(e("ROUTER_TIMEOUT_S", "5.0")),
        max_tokens=int(e("GATEWAY_MAX_TOKENS", "512")),
        cache_enabled=e("GATEWAY_CACHE_ENABLED", "true").lower() in ("1", "true", "yes"),
        cache_size=int(e("GATEWAY_CACHE_SIZE", "500")),
        cache_ttl_s=float(e("GATEWAY_CACHE_TTL_S", "3600")),
        cors_origins=["*"] if origins.strip() == "*" else [o.strip() for o in origins.split(",") if o.strip()],
        cost_per_1k_low=float(e("COST_PER_1K_LOW", "0.0")),
        cost_per_1k_high=float(e("COST_PER_1K_HIGH", "0.0")),
        api_key=e("GATEWAY_API_KEY", ""),
        rate_limit_per_min=int(e("GATEWAY_RATE_LIMIT_PER_MIN", "30")),
    )
