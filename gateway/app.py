"""Backend Gateway (Phase 3 & MVP 2 Phase 2A/2B)."""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import AsyncIterator, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator
from sqlalchemy.orm import Session
from datetime import datetime, timezone
from gateway.cache import ResponseCache
from gateway.logging_utils import JsonlLogger, RequestLog, now_id
from gateway.router_loader import load_router
from gateway.security import make_guard
from gateway.settings import load_gateway_settings
from gateway.tier_clients import StreamChunk, collect_stream, make_high_stream_client, make_low_stream_client
from smartrouter.labels import HIGH, LOW

from gateway.db import database as db_module
from gateway.db.database import get_db
from gateway.db.models import User, Policy, UserPolicy, RefreshToken
from gateway.auth import get_password_hash, verify_password, create_access_token, get_current_user, create_refresh_token, hash_token
from gateway.policy import PolicyEngine

# Alembic (migrations/) is now the source of truth for schema CHANGES in
# production - run `alembic upgrade head` as part of deploy, not this line.
# create_all() is kept only as a dev/test convenience: it's a no-op against a
# database Alembic already migrated (it never alters existing tables), and
# it's what lets tests/test_gateway.py and tests/test_integration_policy.py
# stand up their own throwaway in-memory SQLite databases without needing to
# run the whole migration chain for every test run.
db_module.Base.metadata.create_all(bind=db_module.engine)

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)
api_guard = make_guard(cfg.api_key or None, cfg.rate_limit_per_min)

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
    downgraded: bool = False

class UserCreate(BaseModel):
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str
    
class RefreshRequest(BaseModel):
    refresh_token: str

# --- Auth Endpoints ---
_seeded_engines: set[int] = set()

def seed_default_policies(db: Session):
    # Keyed by the engine's identity (not a bare process-wide flag) so each
    # distinct database - the one real prod engine, and each test file's own
    # throwaway in-memory engine - gets seeded exactly once on its own first
    # call, instead of re-running 3 SELECTs + a commit on every single
    # /api/auth/register request forever after the policies already exist.
    engine_key = id(db.get_bind())
    if engine_key in _seeded_engines:
        return
    policies = [
        {"name": "free", "query_threshold": 50, "action_on_exhaustion": "downgrade_to_low"},
        {"name": "trusted", "query_threshold": 1000, "action_on_exhaustion": "downgrade_to_low"},
        {"name": "admin", "query_threshold": None, "action_on_exhaustion": "allow_overage"}
    ]
    for p in policies:
        if not db.query(Policy).filter(Policy.name == p["name"]).first():
            db.add(Policy(**p))
    db.commit()
    _seeded_engines.add(engine_key)

@app.post("/api/auth/register")
def register(user: UserCreate, db: Session = Depends(get_db)):
    seed_default_policies(db)
    if db.query(User).filter(User.email == user.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed_password = get_password_hash(user.password)
    new_user = User(email=user.email, password_hash=hashed_password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    free_policy = db.query(Policy).filter(Policy.name == "free").first()
    db.add(UserPolicy(user_id=new_user.id, policy_id=free_policy.id))
    db.commit()
    return {"message": "User created successfully"}

@app.post("/api/auth/login")
def login(user: UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()
    if not db_user or not verify_password(user.password, db_user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    access_token = create_access_token(data={"sub": str(db_user.id)})
    refresh_token = create_refresh_token(db, db_user.id)
    return {"access_token": access_token, "refresh_token": refresh_token, "token_type": "bearer"}
@app.post("/api/auth/refresh")
def refresh_access_token(req: RefreshRequest, db: Session = Depends(get_db)):
    invalid = HTTPException(status_code=401, detail="Invalid or expired refresh token")

    token_hash = hash_token(req.refresh_token)
    rt = db.query(RefreshToken).filter(RefreshToken.token_hash == token_hash).first()
    if not rt:
        raise invalid
    if rt.revoked_at is not None:
        raise invalid
    expires_at = rt.expires_at
    if expires_at is not None:
        # SQLite returns this naive; Postgres/Neon returns it tz-aware. Normalize
        # before comparing so this doesn't crash only in production.
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        if expires_at < datetime.now(timezone.utc):
            raise invalid

    access_token = create_access_token(data={"sub": str(rt.user_id)})
    return {"access_token": access_token, "token_type": "bearer"}
@app.post("/api/auth/logout")
def logout(req: RefreshRequest, db: Session = Depends(get_db)):
    # Idempotent and silent either way (unknown token, already-revoked token, or a
    # freshly-revoked one all respond identically) so this endpoint never leaks
    # whether a given refresh token string is/was valid.
    token_hash = hash_token(req.refresh_token)
    rt = db.query(RefreshToken).filter(RefreshToken.token_hash == token_hash).first()
    if rt and rt.revoked_at is None:
        rt.revoked_at = datetime.now(timezone.utc)
        db.commit()
    return {"message": "Logged out"}

# --- Routing & Logging ---
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


# --- Core Endpoints ---
@app.get("/health")
async def health():
    ok = router_status["status"] == "ok" or cfg.router_service_url != ""
    return {"status": "ok" if ok else "degraded"}

@app.post("/api/chat", response_model=ChatResponse, dependencies=[Depends(api_guard)])
async def chat(req: ChatRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    t0 = time.perf_counter()
    request_id = now_id()
    mode = req.mode or cfg.router_mode

    # 1. Policy Evaluation
    policy_engine = PolicyEngine(db)
    policy_decision = policy_engine.evaluate(current_user.id)
    
    if policy_decision.action == "block":
        raise HTTPException(status_code=402, detail=policy_decision.reason)

    # 2. Routing (Bypass if downgraded)
    if policy_decision.is_downgraded:
        decision = {"tier": LOW, "p_strong": 0.0, "threshold": 0.0, "confidence": 1.0, 
                    "mode": mode, "source": "policy_engine", "reasoning": policy_decision.reason}
        router_ms = 0.0
        tier = LOW
    else:
        decision, router_ms = await get_decision(req.query, mode, req.threshold, req.explain)
        tier = req.force_tier if req.force_tier in (LOW, HIGH) else decision["tier"]

    # 3. Cache & Generation
    cache_hit = False
    if cache and req.use_cache:
        hit = cache.get(req.query, mode)
        if hit:
            total = _log_and_finish(request_id, req.query, decision, True, router_ms, 0.0, None, t0, 0)
            return ChatResponse(request_id=request_id, answer=hit.answer,
                                decision=RouteDecisionOut(**{k: decision[k] for k in RouteDecisionOut.model_fields}),
                                cache_hit=True, router_latency_ms=round(router_ms, 1), generation_latency_ms=0.0,
                                total_latency_ms=round(total, 1), tokens_est=0, est_cost_usd=0.0,
                                downgraded=policy_decision.is_downgraded)

    client = tier_client(tier)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": req.query}]
    stats = await collect_stream(client.stream_chat(messages, max_tokens=req.max_tokens or cfg.max_tokens))
    
    if stats.error:
        _log_and_finish(request_id, req.query, decision, False, router_ms, stats.latency_ms, None, t0, 0, stats.error)
        raise HTTPException(status_code=502, detail=f"{tier} tier error: {stats.error}")

    # 4. Record Usage and Return
    cost = cost_of(tier, stats.tokens_est)
    policy_engine.record_usage(current_user.id, tier, stats.tokens_est, cost, policy_decision.is_downgraded)
    
    if cache and req.use_cache:
        cache.put(req.query, mode, stats.text, tier, decision["p_strong"])
        
    total = _log_and_finish(request_id, req.query, decision, False, router_ms, stats.latency_ms, stats.ttft_ms, t0, stats.tokens_est)
    return ChatResponse(
        request_id=request_id, answer=stats.text,
        decision=RouteDecisionOut(**{k: decision.get(k, 0) for k in RouteDecisionOut.model_fields}),
        cache_hit=False, router_latency_ms=round(router_ms, 1), generation_latency_ms=round(stats.latency_ms, 1),
        total_latency_ms=round(total, 1), tokens_est=stats.tokens_est, est_cost_usd=cost,
        downgraded=policy_decision.is_downgraded
    )

@app.post("/api/chat/stream", dependencies=[Depends(api_guard)])
async def chat_stream(req: ChatRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    t0 = time.perf_counter()
    request_id = now_id()
    mode = req.mode or cfg.router_mode
    
    # 1. Policy Evaluation
    policy_engine = PolicyEngine(db)
    policy_decision = policy_engine.evaluate(current_user.id)
    if policy_decision.action == "block":
        raise HTTPException(status_code=402, detail=policy_decision.reason)

    if policy_decision.is_downgraded:
        decision = {"tier": LOW, "p_strong": 0.0, "threshold": 0.0, "confidence": 1.0, 
                    "mode": mode, "source": "policy_engine", "reasoning": policy_decision.reason}
        router_ms = 0.0
        tier = LOW
    else:
        decision, router_ms = await get_decision(req.query, mode, req.threshold, req.explain)
        tier = req.force_tier if req.force_tier in (LOW, HIGH) else decision["tier"]

    async def gen() -> AsyncIterator[bytes]:
        def sse(event: str, data: dict) -> bytes:
            return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()

        yield sse("meta", {"request_id": request_id, "decision": decision, "cache_hit": False,
                           "router_latency_ms": round(router_ms, 1), "downgraded": policy_decision.is_downgraded})

        client = tier_client(tier)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": req.query}]
        text_parts, ttft_ms, gen_t0, err = [], None, time.perf_counter(), None
        
        async for chunk in client.stream_chat(messages, max_tokens=req.max_tokens or cfg.max_tokens):
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
        cost = cost_of(tier, tokens_est)

        # Record usage via a fresh session context so it persists after StreamingResponse closes the main
        # request. Goes through db_module.SessionLocal at call time (not a bare name copied at import
        # time) so tests that monkeypatch gateway.db.database.SessionLocal are correctly picked up here too.
        if not err:
            with db_module.SessionLocal() as record_db:
                usage_engine = PolicyEngine(record_db)
                usage_engine.record_usage(current_user.id, tier, tokens_est, cost, policy_decision.is_downgraded)

        total = _log_and_finish(request_id, req.query, decision, False, router_ms, gen_ms, ttft_ms, t0, tokens_est, err)
        yield sse("done", {"total_latency_ms": round(total, 1), "cache_hit": False, "tokens_est": tokens_est,
                           "est_cost_usd": cost, "ttft_ms": round(ttft_ms, 1) if ttft_ms else None})

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
