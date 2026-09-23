"""SLMRouter: local small LLM (via Ollama) -> RouteDecision."""
from __future__ import annotations

from .clients import OllamaClient
from .labels import LOW
from .slm_router import OUTPUT_SCHEMA, RouteDecision, build_messages, parse_decision


class SLMRouter:
    def __init__(self, client: OllamaClient, medium_tier: str = LOW):
        self.client = client
        self.medium_tier = medium_tier

    def route(self, query: str) -> RouteDecision:
        res = self.client.chat(
            build_messages(query),
            fmt=OUTPUT_SCHEMA,
            max_tokens=40,      # the JSON answer is ~20 tokens; cap it for latency
            temperature=0.0,
        )
        decision = parse_decision(res.text, self.medium_tier)
        decision.latency_ms = res.latency_ms
        return decision
