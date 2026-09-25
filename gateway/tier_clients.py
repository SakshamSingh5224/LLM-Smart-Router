"""Async, streaming versions of the tier clients used by the gateway.

smartrouter/clients.py is synchronous (fine for CLI scripts/benchmarks). The
gateway is an async FastAPI service that must stream tokens to the browser as
they arrive, so it gets its own async httpx clients here rather than reusing
those.

* OllamaStreamClient        - local model via Ollama (low tier by default)
* OpenAICompatStreamClient - any OpenAI-compatible SSE endpoint (Groq by default,
                             used for the high tier)
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional

import httpx


@dataclass
class StreamChunk:
    delta: str = ""
    done: bool = False
    error: Optional[str] = None


@dataclass
class StreamStats:
    text: str = ""
    latency_ms: float = 0.0
    ttft_ms: Optional[float] = None
    tokens_est: int = 0
    error: Optional[str] = None
    raw_meta: dict = field(default_factory=dict)


def _estimate_tokens(text: str) -> int:
    # ~4 chars/token is a common rough estimate for English text; good enough
    # for a cost dashboard, not for billing.
    return max(1, len(text) // 4)


class OllamaStreamClient:
    provider = "ollama"

    def __init__(self, host: str, model: str, timeout: float = 180.0):
        self.host = host.rstrip("/")
        self.model = model

    async def stream_chat(self, messages: list[dict], *, max_tokens: int = 8192,
                           temperature: float = 0.7) -> AsyncIterator[StreamChunk]:
        payload = {
            "model": self.model, "messages": messages, "stream": True,
            "keep_alive": "30m",
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        async with httpx.AsyncClient(timeout=self._timeout()) as client:
            try:
                async with client.stream("POST", f"{self.host}/api/chat", json=payload) as r:
                    if r.status_code != 200:
                        body = (await r.aread()).decode("utf-8", "ignore")[:300]
                        yield StreamChunk(error=f"ollama HTTP {r.status_code}: {body}")
                        return
                    async for line in r.aiter_lines():
                        if not line.strip():
                            continue
                        data = json.loads(line)
                        piece = data.get("message", {}).get("content", "")
                        if piece:
                            yield StreamChunk(delta=piece)
                        if data.get("done"):
                            yield StreamChunk(done=True)
            except httpx.HTTPError as e:
                yield StreamChunk(error=f"ollama connection error: {e}")

    def _timeout(self) -> httpx.Timeout:
        return httpx.Timeout(180.0, connect=10.0)


class OpenAICompatStreamClient:
    provider = "openai_compat"

    def __init__(self, base_url: str, api_key: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    async def stream_chat(self, messages: list[dict], *, max_tokens: int = 8192,
                           temperature: float = 0.7, retries: int = 2, max_continuations: int = 3) -> AsyncIterator[StreamChunk]:
        
        # Keep track of the full conversation history for continuations
        current_messages = list(messages)
        
        # Allow the model to Auto-Continue up to 'max_continuations' times
        for cont_idx in range(max_continuations + 1):
            payload: dict = {
                "model": self.model, 
                "messages": current_messages, 
                "stream": True,
                "temperature": temperature, 
                "max_tokens": max_tokens,
            }
            if "gpt-oss" in self.model:
                payload["reasoning_effort"] = "low"

            assistant_text_this_round = ""
            finish_reason = None
            request_success = False

            async with httpx.AsyncClient(timeout=None) as client:
                for attempt in range(retries + 1):
                    try:
                        async with client.stream("POST", f"{self.base_url}/chat/completions",
                                                  headers=self._headers, json=payload) as r:
                            
                            # Handle rate limits
                            if r.status_code == 429 and attempt < retries:
                                wait = float(r.headers.get("retry-after", 2 * (attempt + 1)))
                                await self._sleep(min(wait, 20.0))
                                continue
                            
                            # Handle normal errors
                            if r.status_code != 200:
                                body = (await r.aread()).decode("utf-8", "ignore")[:300]
                                yield StreamChunk(error=f"HTTP {r.status_code}: {body}")
                                return
                            
                            # Parse the SSE Stream
                            async for line in r.aiter_lines():
                                line = line.strip()
                                if not line or not line.startswith("data:"):
                                    continue
                                
                                payload_str = line[len("data:"):].strip()
                                
                                if payload_str == "[DONE]":
                                    if not finish_reason:
                                        finish_reason = "stop"
                                    break
                                
                                try:
                                    obj = json.loads(payload_str)
                                except json.JSONDecodeError:
                                    continue
                                
                                choice = (obj.get("choices") or [{}])[0]
                                piece = (choice.get("delta") or {}).get("content") or ""
                                
                                if piece:
                                    assistant_text_this_round += piece
                                    yield StreamChunk(delta=piece)
                                
                                fr = choice.get("finish_reason")
                                if fr:
                                    finish_reason = fr
                            
                            request_success = True
                            break  # Exit the retry loop if successful
                    
                    except httpx.HTTPError as e:
                        if attempt < retries:
                            await self._sleep(2 * (attempt + 1))
                            continue
                        yield StreamChunk(error=f"connection error: {e}")
                        return

            if not request_success:
                return

            # --- THE AUTO-CONTINUE LOGIC ---
            # If the API cut us off due to length, and we haven't exceeded our continuation limit:
            if finish_reason in ("length", "max_tokens") and cont_idx < max_continuations:
                # 1. Save what the assistant generated in this round
                current_messages.append({"role": "assistant", "content": assistant_text_this_round})
                
                # 2. Add a hidden system prompt instructing it to seamlessly resume
                current_messages.append({
                    "role": "user", 
                    "content": "Your response was cut off due to length. Please continue EXACTLY from where you left off. Do not include any introductory or transition text, just provide the very next word or character."
                })
                # The loop will now restart, firing a new API request invisibly.
            else:
                # We finished naturally, or hit the absolute continuation limit
                yield StreamChunk(done=True)
                return

    @staticmethod
    async def _sleep(s: float) -> None:
        import asyncio
        await asyncio.sleep(s)


async def collect_stream(chunks: AsyncIterator[StreamChunk]) -> StreamStats:
    """Drain a stream into a single StreamStats (used by non-streaming callers/tests)."""
    text, t0, ttft, err = [], time.perf_counter(), None, None
    async for c in chunks:
        if c.error:
            err = c.error
            break
        if c.delta:
            if ttft is None:
                ttft = (time.perf_counter() - t0) * 1000
            text.append(c.delta)
        if c.done:
            break
    full = "".join(text)
    return StreamStats(full, (time.perf_counter() - t0) * 1000, ttft, _estimate_tokens(full), err)


def make_low_stream_client(cfg):
    if cfg.low_provider == "ollama":
        return OllamaStreamClient(cfg.ollama_host, cfg.low_model)
    return OpenAICompatStreamClient(cfg.low_base_url, cfg.low_api_key, cfg.low_model)


def make_high_stream_client(cfg) -> OpenAICompatStreamClient:
    return OpenAICompatStreamClient(cfg.high_base_url, cfg.high_api_key, cfg.high_model)
