"""Backend Gateway (Phase 3).

Wires up the full request path from user input to final response:
  1. accept a query from the frontend
  2. get a tier decision (in-process router, or the router_service over HTTP
     if ROUTER_SERVICE_URL is set)
  3. forward the query to the selected model tier (low-cost or high-cost)
  4. stream the response back to the frontend over Server-Sent Events
  5. log routing decisions, latencies, and estimated cost
  6. serve the static frontend

Run:
    uvicorn gateway.app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import AsyncIterator, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import StreamingResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from gateway.cache import ResponseCache  # noqa: E402
from gateway.logging_utils import JsonlLogger, RequestLog, now_id  # noqa: E402
from gateway.router_loader import load_router  # noqa: E402
from gateway.settings import load_gateway_settings  # noqa: E402
from gateway.tier_clients import StreamChunk, collect_stream, make_high_stream_client, make_low_stream_client  # noqa: E402
from smartrouter.labels import HIGH, LOW  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("gateway")

cfg = load_gateway_settings()
router, router_status = load_router(cfg.router_artifact_path)
log.info("Router status: %s", router_status)

low_client = make_low_stream_client(cfg.model)
high_client = make_high_stream_client(cfg.model)
cache = ResponseCache(cfg.cache_size, cfg.cache_ttl_s) if cfg.cache_enabled else None
jlog = JsonlLogger(cfg.log_path)
_http = httpx.AsyncClient(timeout=cfg.router_timeout_s)

app = FastAPI(title="LLM Smart Router - Gateway", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=cfg.cors_origins, allow_methods=["*"], allow_headers=["*"])

SYSTEM_PROMPT = "You are a helpful, concise assistant."


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=20000)
    mode: Optional[str] = Field(None, description="aggressive|balanced|conservative")
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    force_tier: Optional[str] = Field(None, description="Debug override: 'low' or 'high'")
    max_tokens: Optional[int] = Field(None, ge=1, le=4096)
    use_cache: bool = True
    explain: bool = False


class RouteDecisionOut(BaseModel):
    tier: str
    p_strong: float
    threshold: float
    confidence: float
    mode: str
    source: str
    reasoning: Optional[str] = None


class ChatResponse(BaseModel):
    request_id: str
    answer: str
    decision: RouteDecisionOut
    cache_hit: bool
    router_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    tokens_est: int
    est_cost_usd: float


# --------------------------------------------------------------------------
# Routing: in-process, or delegate to router_service if ROUTER_SERVICE_URL set
# --------------------------------------------------------------------------
async def get_decision(query: str, mode, threshold, explain: bool):
    t0 = time.perf_counter()
    if cfg.router_service_url:
        try:
            r = await _http.post(f"{cfg.router_service_url}/route",
                                  json={"query": query, "mode": mode, "threshold": threshold, "explain": explain})
            r.raise_for_status()
            d = r.json()
            return d, (time.perf_counter() - t0) * 1000
        except (httpx.HTTPError, ValueError) as e:
            log.warning("router_service unreachable (%s) - falling back to in-process rules", e)
            from smartrouter.classifier_router import RulesRouter

            d = RulesRouter().route(query, explain=explain)
    else:
        try:
            d = router.route(query, mode=mode, threshold=threshold, explain=explain)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
    return {"tier": d.tier, "p_strong": d.p_strong, "threshold": d.threshold, "confidence": d.confidence,
            "mode": d.mode, "source": d.source, "reasoning": d.reasoning,
            "signals": getattr(d, "signals", [])}, (time.perf_counter() - t0) * 1000


def tier_client(tier: str):
    return low_client if tier == LOW else high_client


def cost_of(tier: str, tokens_est: int) -> float:
    rate = cfg.cost_per_1k_low if tier == LOW else cfg.cost_per_1k_high
    return round(rate * tokens_est / 1000.0, 6)


def _log_and_finish(request_id, query, decision, cache_hit, router_ms, gen_ms, ttft_ms, t_start, tokens_est, err=None):
    total_ms = (time.perf_counter() - t_start) * 1000
    jlog.write(RequestLog(
        request_id=request_id, ts=time.time(), query_chars=len(query), mode=decision["mode"],
        tier=decision["tier"], source=decision["source"], p_strong=decision["p_strong"],
        confidence=decision["confidence"], cache_hit=cache_hit, router_latency_ms=round(router_ms, 1),
        generation_latency_ms=round(gen_ms, 1), ttft_ms=round(ttft_ms, 1) if ttft_ms else None,
        total_latency_ms=round(total_ms, 1), tokens_est=tokens_est,
        est_cost_usd=cost_of(decision["tier"], tokens_est), error=err, signals=decision.get("signals", []),
    ))
    return total_ms


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------
@app.get("/health")
async def health():
    ok = router_status["status"] == "ok" or cfg.router_service_url != ""
    return {"status": "ok" if ok else "degraded", "router": router_status,
            "router_service_url": cfg.router_service_url or None, "cache": cache.stats() if cache else None}


@app.get("/modes")
def modes():
    if cfg.router_service_url:
        return {"note": "modes are served by the router_service", "url": f"{cfg.router_service_url}/modes"}
    return {"default_mode": getattr(router, "default_mode", cfg.router_mode), "modes": getattr(router, "modes", {})}


@app.get("/logs/recent")
def recent_logs(n: int = 20):
    return jlog.tail(min(n, 200))


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Non-streaming endpoint: full answer in one JSON response."""
    t0 = time.perf_counter()
    request_id = now_id()
    mode = req.mode or cfg.router_mode

    decision, router_ms = await get_decision(req.query, mode, req.threshold, req.explain)
    tier = req.force_tier if req.force_tier in (LOW, HIGH) else decision["tier"]

    cache_hit = False
    if cache and req.use_cache:
        hit = cache.get(req.query, mode)
        if hit:
            cache_hit = True
            total = _log_and_finish(request_id, req.query, decision, True, router_ms, 0.0, None, t0, 0)
            return ChatResponse(request_id=request_id, answer=hit.answer,
                                decision=RouteDecisionOut(**{k: decision[k] for k in RouteDecisionOut.model_fields}),
                                cache_hit=True, router_latency_ms=round(router_ms, 1), generation_latency_ms=0.0,
                                total_latency_ms=round(total, 1), tokens_est=0, est_cost_usd=0.0)

    client = tier_client(tier)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": req.query}]
    max_tokens = req.max_tokens or cfg.max_tokens
    stats = await collect_stream(client.stream_chat(messages, max_tokens=max_tokens))
    if stats.error:
        total = _log_and_finish(request_id, req.query, decision, False, router_ms, stats.latency_ms, None, t0, 0, stats.error)
        raise HTTPException(status_code=502, detail=f"{tier} tier error: {stats.error}")

    if cache and req.use_cache:
        cache.put(req.query, mode, stats.text, tier, decision["p_strong"])
    total = _log_and_finish(request_id, req.query, decision, False, router_ms, stats.latency_ms, stats.ttft_ms,
                            t0, stats.tokens_est)
    return ChatResponse(
        request_id=request_id, answer=stats.text,
        decision=RouteDecisionOut(**{k: decision[k] for k in RouteDecisionOut.model_fields}),
        cache_hit=False, router_latency_ms=round(router_ms, 1), generation_latency_ms=round(stats.latency_ms, 1),
        total_latency_ms=round(total, 1), tokens_est=stats.tokens_est, est_cost_usd=cost_of(tier, stats.tokens_est),
    )


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    """SSE endpoint: `meta` event with the routing decision, then `delta` events per
    token chunk, then a final `done` event with latency/cost totals."""
    t0 = time.perf_counter()
    request_id = now_id()
    mode = req.mode or cfg.router_mode
    decision, router_ms = await get_decision(req.query, mode, req.threshold, req.explain)
    tier = req.force_tier if req.force_tier in (LOW, HIGH) else decision["tier"]

    async def gen() -> AsyncIterator[bytes]:
        def sse(event: str, data: dict) -> bytes:
            return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()

        yield sse("meta", {"request_id": request_id, "decision": decision, "cache_hit": False,
                           "router_latency_ms": round(router_ms, 1)})

        if cache and req.use_cache:
            hit = cache.get(req.query, mode)
            if hit:
                for i in range(0, len(hit.answer), 40):  # replay cached text in small chunks
                    yield sse("delta", {"text": hit.answer[i:i + 40]})
                    await asyncio.sleep(0)
                total = _log_and_finish(request_id, req.query, decision, True, router_ms, 0.0, None, t0, 0)
                yield sse("done", {"total_latency_ms": round(total, 1), "cache_hit": True, "tokens_est": 0,
                                   "est_cost_usd": 0.0})
                return

        client = tier_client(tier)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": req.query}]
        max_tokens = req.max_tokens or cfg.max_tokens
        text_parts, ttft_ms, gen_t0, err = [], None, time.perf_counter(), None
        async for chunk in client.stream_chat(messages, max_tokens=max_tokens):
            if isinstance(chunk, StreamChunk) and chunk.error:
                err = chunk.error
                yield sse("error", {"message": err})
                break
            if chunk.delta:
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - gen_t0) * 1000
                text_parts.append(chunk.delta)
                yield sse("delta", {"text": chunk.delta})
            if chunk.done:
                break
        gen_ms = (time.perf_counter() - gen_t0) * 1000
        full_text = "".join(text_parts)
        tokens_est = max(1, len(full_text) // 4) if full_text else 0

        if not err and cache and req.use_cache and full_text:
            cache.put(req.query, mode, full_text, tier, decision["p_strong"])
        total = _log_and_finish(request_id, req.query, decision, False, router_ms, gen_ms, ttft_ms, t0, tokens_est, err)
        yield sse("done", {"total_latency_ms": round(total, 1), "cache_hit": False, "tokens_est": tokens_est,
                           "est_cost_usd": cost_of(tier, tokens_est), "ttft_ms": round(ttft_ms, 1) if ttft_ms else None})

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# Serve the frontend (built with no framework, so nothing to compile) if present.
_frontend_dir = Path(__file__).resolve().parents[1] / "frontend"
if _frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")
