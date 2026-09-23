"""Thin HTTP clients for the three model endpoints used in Phase 1.

* OllamaClient        - local SLM (router + optional low tier)
* OpenAICompatClient  - any OpenAI-compatible API (Groq free tier by default)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import httpx


@dataclass
class ChatResult:
    text: str
    latency_ms: float
    model: str
    raw: dict = field(default_factory=dict)


class OllamaClient:
    def __init__(self, host: str, model: str, timeout: float = 180.0):
        self.host = host.rstrip("/")
        self.model = model
        self._http = httpx.Client(timeout=timeout)

    # -- health ---------------------------------------------------------
    def is_up(self) -> bool:
        try:
            return self._http.get(f"{self.host}/api/tags", timeout=5.0).status_code == 200
        except httpx.HTTPError:
            return False

    def installed_models(self) -> list[str]:
        r = self._http.get(f"{self.host}/api/tags", timeout=10.0)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]

    def has_model(self) -> bool:
        names = self.installed_models()
        return self.model in names or f"{self.model}:latest" in names

    # -- inference --------------------------------------------------------
    def chat(
        self,
        messages: list[dict],
        *,
        fmt: Optional[dict] = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
        keep_alive: str = "30m",
    ) -> ChatResult:
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": keep_alive,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if fmt is not None:
            payload["format"] = fmt
        t0 = time.perf_counter()
        r = self._http.post(f"{self.host}/api/chat", json=payload)
        r.raise_for_status()
        ms = (time.perf_counter() - t0) * 1000
        data = r.json()
        return ChatResult(data.get("message", {}).get("content", ""), ms, self.model, data)

    def unload(self) -> None:
        """Evict the model from memory so the next call is a true cold start."""
        self._http.post(
            f"{self.host}/api/generate",
            json={"model": self.model, "keep_alive": 0},
            timeout=30.0,
        )
        time.sleep(1.5)


class OpenAICompatClient:
    """Minimal /chat/completions client with 429 back-off (free tiers rate-limit)."""

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 90.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        self._http = httpx.Client(timeout=timeout)

    def list_models(self) -> list[str]:
        r = self._http.get(f"{self.base_url}/models", headers=self._headers, timeout=15.0)
        r.raise_for_status()
        return sorted(m["id"] for m in r.json().get("data", []))

    def chat(
        self,
        messages: list[dict],
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        retries: int = 3,
    ) -> ChatResult:
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
        }
        if "gpt-oss" in self.model:
            payload["reasoning_effort"] = "low"  # reasoning models: keep it cheap/fast

        t0 = time.perf_counter()
        for attempt in range(retries + 1):
            r = self._http.post(f"{self.base_url}/chat/completions", headers=self._headers, json=payload)
            if r.status_code == 429 and attempt < retries:
                wait = float(r.headers.get("retry-after", 2 * (attempt + 1)))
                time.sleep(min(wait, 30.0))
                continue
            r.raise_for_status()
            break
        ms = (time.perf_counter() - t0) * 1000
        data = r.json()
        text = (data["choices"][0]["message"].get("content") or "").strip()
        return ChatResult(text, ms, self.model, data)


def make_low_client(cfg):
    if cfg.low_provider == "ollama":
        return OllamaClient(cfg.ollama_host, cfg.low_model)
    return OpenAICompatClient(cfg.low_base_url, cfg.low_api_key, cfg.low_model)


def make_high_client(cfg) -> OpenAICompatClient:
    return OpenAICompatClient(cfg.high_base_url, cfg.high_api_key, cfg.high_model)
