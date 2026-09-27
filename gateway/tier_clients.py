"""Async, streaming versions of the tier clients used by the gateway.

smartrouter/clients.py is synchronous (fine for CLI scripts/benchmarks). The
gateway is an async FastAPI service that must stream tokens to the browser as
they arrive, so it gets its own async httpx clients here rather than reusing
those.

* OllamaStreamClient       - local model via Ollama (low tier by default)
* OpenAICompatStreamClient - any OpenAI-compatible SSE endpoint (Groq by default,
                              used for the high tier)

Both auto-continue when the upstream stops early only because it hit its token
cap (finish_reason/done_reason == "length"): they transparently open a fresh
upstream stream with the partial answer fed back as context, and keep yielding
StreamChunks to the caller as one continuous stream. The caller (gateway/app.py)
never sees the seam - it only sees `done=True` once the model actually finished.

IMPORTANT: as soon as the real finish_reason is seen, _stream_once returns
immediately. Groq (like OpenAI) sends a trailing `data: [DONE]` line after the
line that carries finish_reason - if we kept reading past it, that sentinel
would yield a second StreamChunk(done=True, finish_reason="stop") that silently
overwrites a real "length", which is exactly what broke auto-continue before.
"""
from __future__ import annotations

import asyncio
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
    finish_reason: Optional[str] = None  # "stop" | "length" | ... ; None while still streaming


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


CONTINUE_NUDGE = "Continue exactly where you left off. Do not repeat anything, do not add any preamble."
MAX_CONTINUATIONS = 6


class OllamaStreamClient:
    provider = "ollama"

    def __init__(self, host: str, model: str, timeout: float = 180.0):
        self.host = host.rstrip("/")
        self.model = model

    async def _stream_once(self, messages: list[dict], max_tokens: int, temperature: float) -> AsyncIterator[StreamChunk]:
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
                            yield StreamChunk(done=True, finish_reason=data.get("done_reason") or "stop")
                            return  # Ollama's own final line - nothing meaningful follows it
            except httpx.HTTPError as e:
                yield StreamChunk(error=f"ollama connection error: {e}")

    async def stream_chat(self, messages: list[dict], *, max_tokens: int = 512,
                           temperature: float = 0.7,
                           max_continuations: int = MAX_CONTINUATIONS) -> AsyncIterator[StreamChunk]:
        working = list(messages)
        accumulated = ""
        for _ in range(max_continuations + 1):
            finish_reason = None
            async for c in self._stream_once(working, max_tokens, temperature):
                if c.error:
                    yield c
                    return
                if c.delta:
                    accumulated += c.delta
                    yield StreamChunk(delta=c.delta)
                if c.done:
                    finish_reason = c.finish_reason
            if finish_reason != "length":
                yield StreamChunk(done=True, finish_reason=finish_reason)
                return
            working = messages + [
                {"role": "assistant", "content": accumulated},
                {"role": "user", "content": CONTINUE_NUDGE},
            ]
        yield StreamChunk(delta="\n\n[response truncated after max continuations]")
        yield StreamChunk(done=True, finish_reason="length")

    def _timeout(self) -> httpx.Timeout:
        return httpx.Timeout(180.0, connect=10.0)


class OpenAICompatStreamClient:
    provider = "openai_compat"

    def __init__(self, base_url: str, api_key: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    async def _stream_once(self, messages: list[dict], max_tokens: int, temperature: float,
                            retries: int) -> AsyncIterator[StreamChunk]:
        payload: dict = {
            "model": self.model, "messages": messages, "stream": True,
            "temperature": temperature, "max_completion_tokens": max_tokens,
        }
        if "gpt-oss" in self.model:
            payload["reasoning_effort"] = "low"

        async with httpx.AsyncClient(timeout=httpx.Timeout(90.0, connect=10.0)) as client:
            for attempt in range(retries + 1):
                try:
                    async with client.stream("POST", f"{self.base_url}/chat/completions",
                                              headers=self._headers, json=payload) as r:
                        if r.status_code in (429, 413) and attempt < retries:
                            # 429 = per-request rate limit; 413 here = Groq's TPM (tokens-per-minute)
                            # cap tripped. Neither has a useful retry-after for 413, so back off long
                            # enough for the rolling 60s window to free up capacity.
                            wait = float(r.headers.get("retry-after", 15 * (attempt + 1)))
                            await asyncio.sleep(min(wait, 45.0))
                            continue
                        if r.status_code != 200:
                            body = (await r.aread()).decode("utf-8", "ignore")[:300]
                            yield StreamChunk(error=f"HTTP {r.status_code}: {body}")
                            return
                        async for line in r.aiter_lines():
                            line = line.strip()
                            if not line or not line.startswith("data:"):
                                continue
                            payload_str = line[len("data:"):].strip()
                            if payload_str == "[DONE]":
                                # only reached if a finish_reason was never seen on an earlier line
                                yield StreamChunk(done=True, finish_reason="stop")
                                return
                            try:
                                obj = json.loads(payload_str)
                            except json.JSONDecodeError:
                                continue
                            choice = (obj.get("choices") or [{}])[0]
                            piece = (choice.get("delta") or {}).get("content") or ""
                            if piece:
                                yield StreamChunk(delta=piece)
                            fr = choice.get("finish_reason")
                            if fr:
                                yield StreamChunk(done=True, finish_reason=fr)
                                return  # stop immediately - do NOT keep reading into a trailing [DONE]
                        return
                except httpx.HTTPError as e:
                    if attempt < retries:
                        await asyncio.sleep(2 * (attempt + 1))
                        continue
                    yield StreamChunk(error=f"connection error: {e}")
                    return

    async def stream_chat(self, messages: list[dict], *, max_tokens: int = 512,
                           temperature: float = 0.7, retries: int = 2,
                           max_continuations: int = MAX_CONTINUATIONS) -> AsyncIterator[StreamChunk]:
        working = list(messages)
        accumulated = ""
        for _ in range(max_continuations + 1):
            finish_reason = None
            async for c in self._stream_once(working, max_tokens, temperature, retries):
                if c.error:
                    yield c
                    return
                if c.delta:
                    accumulated += c.delta
                    yield StreamChunk(delta=c.delta)
                if c.done:
                    finish_reason = c.finish_reason
            if finish_reason != "length":
                yield StreamChunk(done=True, finish_reason=finish_reason)
                return
            # truncated only because of the token cap: feed the partial answer back and keep going
            working = messages + [
                {"role": "assistant", "content": accumulated},
                {"role": "user", "content": CONTINUE_NUDGE},
            ]
        yield StreamChunk(delta="\n\n[response truncated after max continuations]")
        yield StreamChunk(done=True, finish_reason="length")


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
