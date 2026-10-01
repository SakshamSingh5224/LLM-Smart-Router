"""Thin HTTP clients for the model endpoints.

* OllamaClient        - local SLM (router + optional local low tier)
* OpenAICompatClient  - any OpenAI-compatible API (Groq, used for both low and high tiers here)
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

    # -- inference (with auto-continue on truncation) --------------------
    def _request(self, messages: list[dict], fmt, max_tokens: int, temperature: float, keep_alive: str):
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": keep_alive,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if fmt is not None:
            payload["format"] = fmt
        r = self._http.post(f"{self.host}/api/chat", json=payload)
        r.raise_for_status()
        data = r.json()
        text = data.get("message", {}).get("content", "")
        done_reason = data.get("done_reason")  # "stop" | "length" | ...
        return text, done_reason, data

    def chat(
        self,
        messages: list[dict],
        *,
        fmt: Optional[dict] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        keep_alive: str = "30m",
        max_continuations: int = 6,
    ) -> ChatResult:
        working = list(messages)
        full_text = ""
        last_raw: dict = {}
        t0 = time.perf_counter()

        for _ in range(max_continuations + 1):
            chunk, done_reason, last_raw = self._request(working, fmt, max_tokens, temperature, keep_alive)
            full_text += chunk
            if done_reason != "length":
                break
            working = working + [
                {"role": "assistant", "content": chunk},
                {"role": "user", "content": "Continue exactly where you left off. "
                                             "Do not repeat anything, do not add any preamble."},
            ]
        else:
            full_text += "\n\n[response truncated after max continuations]"

        ms = (time.perf_counter() - t0) * 1000
        return ChatResult(full_text.strip() if fmt is None else full_text, ms, self.model, last_raw)

    def unload(self) -> None:
        """Evict the model from memory so the next call is a true cold start."""
        self._http.post(
            f"{self.host}/api/generate",
            json={"model": self.model, "keep_alive": 0},
            timeout=30.0,
        )
        time.sleep(1.5)


class OpenAICompatClient:
    """Minimal /chat/completions client with 429 back-off and auto-continue on truncation.

    Used for both the low tier and the high tier here, since both point at Groq.
    """

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 90.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        self._http = httpx.Client(timeout=timeout)

    def list_models(self) -> list[str]:
        r = self._http.get(f"{self.base_url}/models", headers=self._headers, timeout=15.0)
        r.raise_for_status()
        return sorted(m["id"] for m in r.json().get("data", []))

    def _request(self, messages: list[dict], max_tokens: int, temperature: float, retries: int):
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
        }
        if "gpt-oss" in self.model:
            payload["reasoning_effort"] = "low"  # reasoning models: keep it cheap/fast

        for attempt in range(retries + 1):
            r = self._http.post(f"{self.base_url}/chat/completions", headers=self._headers, json=payload)
            if r.status_code == 429 and attempt < retries:
                wait = float(r.headers.get("retry-after", 2 * (attempt + 1)))
                time.sleep(min(wait, 30.0))
                continue
            r.raise_for_status()
            break
        data = r.json()
        choice = data["choices"][0]
        text = choice["message"].get("content") or ""
        finish_reason = choice.get("finish_reason")  # "stop" | "length" | ...
        return text, finish_reason, data

    def chat(
        self,
        messages: list[dict],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        retries: int = 3,
        max_continuations: int = 6,
    ) -> ChatResult:
        working = list(messages)
        full_text = ""
        last_raw: dict = {}
        t0 = time.perf_counter()

        for _ in range(max_continuations + 1):
            chunk, finish_reason, last_raw = self._request(working, max_tokens, temperature, retries)
            full_text += chunk
            if finish_reason != "length":
                break  # finished naturally - done
            working = working + [
                {"role": "assistant", "content": chunk},
                {"role": "user", "content": "Continue exactly where you left off. "
                                             "Do not repeat anything, do not add any preamble."},
            ]
        else:
            full_text += "\n\n[response truncated after max continuations]"

        ms = (time.perf_counter() - t0) * 1000
        return ChatResult(full_text.strip(), ms, self.model, last_raw)


def make_low_client(cfg):
    if cfg.low_provider == "ollama":
        return OllamaClient(cfg.ollama_host, cfg.low_model)
    return OpenAICompatClient(cfg.low_base_url, cfg.low_api_key, cfg.low_model)


def make_high_client(cfg) -> OpenAICompatClient:
    return OpenAICompatClient(cfg.high_base_url, cfg.high_api_key, cfg.high_model)
