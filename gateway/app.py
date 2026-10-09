"""Backend Gateway (Phase 3 & MVP 2 Phase 2A/2B)."""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import AsyncIterator, List, Optional

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
from gateway.tier_clients import (
    OllamaStreamClient, StreamChunk, collect_stream, make_high_stream_client, make_low_stream_client,
)
from gateway.rag_client import LOCAL_RAG, LocalRagClient, RagRequest
from smartrouter.labels import HIGH, LOW

from gateway.db import database as db_module
from gateway.db.database import get_db
from gateway.db.models import User, Policy, UserPolicy, RefreshToken
from gateway.auth import get_password_hash, verify_password, create_access_token, get_current_user, create_refresh_token, hash_token
from gateway.policy import PolicyEngine
from gateway.metrics import (
    POLICY_DECISION_COUNTER, ROUTED_TO_LOCAL, EXTERNAL_FALLBACK, CACHE_HIT,
    RETRIEVAL_LATENCY, RERANK_LATENCY, LOCAL_GENERATION_LATENCY, LOCAL_TTFT,
    ESTIMATED_COST_AVOIDED,
)
from gateway.judge import QueryJudge
from gateway.reranker import RetrievalPipeline

db_module.Base.metadata.create_all(bind=db_module.engine)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("gateway")

cfg = load_gateway_settings()
router, router_status = load_router(cfg.router_artifact_path)
log.info("Router status: %s", router_status)

judge = QueryJudge()
retrieval_pipeline = RetrievalPipeline()

low_client = make_low_stream_client(cfg.model)
high_client = make_high_stream_client(cfg.model)
local_rag_client = LocalRagClient(OllamaStreamClient(cfg.model.ollama_host, cfg.local_rag_model))
cache = (
    ResponseCache(
        cfg.cache_size,
        cfg.cache_ttl_s,
        redis_url=cfg.redis_url,
        semantic_enabled=cfg.semantic_cache_enabled,
        semantic_threshold=cfg.semantic_cache_threshold,
        embedding_model=cfg.embedding_model,
    )
    if cfg.cache_enabled
    else None
)

# Phase 3E: warm the semantic embedding model before the gateway
# starts serving requests. This keeps model initialization out of
# semantic-cache hit latency.
if cache is not None and cfg.semantic_cache_enabled:
    cache.warm_semantic_model()

jlog = JsonlLogger(cfg.log_path)
_http = httpx.AsyncClient(timeout=cfg.router_timeout_s)

app = FastAPI(title="LLM Smart Router - Gateway", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=cfg.cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key"],
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
    ttft_ms: Optional[float] = None
    estimated_cost_avoided_usd: float = 0.0
    sources: list[dict] = Field(default_factory=list)

class UserCreate(BaseModel):
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str
    
class RefreshRequest(BaseModel):
    refresh_token: str

class PolicyOut(BaseModel):
    model_config = {"from_attributes": True}
    id: int
    name: str
    tier_scope: str
    query_threshold: Optional[int] = None
    token_threshold: Optional[int] = None
    period: str
    action_on_exhaustion: str

class PolicyCreate(BaseModel):
    name: str
    tier_scope: str = "all"
    query_threshold: Optional[int] = None
    token_threshold: Optional[int] = None
    period: str = "monthly"
    action_on_exhaustion: str = "downgrade_to_low"

class PolicyUpdate(BaseModel):
    tier_scope: Optional[str] = None
    query_threshold: Optional[int] = None
    token_threshold: Optional[int] = None
    period: Optional[str] = None
    action_on_exhaustion: Optional[str] = None

class UserAdminOut(BaseModel):
    id: int
    email: str
    is_admin: bool
    policy_id: Optional[int] = None
    policy_name: Optional[str] = None
    queries_used: int
    tokens_used: int

class ReassignPolicyRequest(BaseModel):
    policy_id: int

class UsageOut(BaseModel):
    policy_name: str
    tier_scope: str
    period: str
    action_on_exhaustion: str
    query_threshold: Optional[int] = None
    token_threshold: Optional[int] = None
    queries_used: int
    tokens_used: int


def require_admin(current_user: User = Depends(get_current_user)) -> User:
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user


# --- Auth Endpoints ---
_seeded_engines: set[int] = set()

def seed_default_policies(db: Session):
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
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        if expires_at < datetime.now(timezone.utc):
            raise invalid

    access_token = create_access_token(data={"sub": str(rt.user_id)})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/api/auth/logout")
def logout(req: RefreshRequest, db: Session = Depends(get_db)):
    token_hash = hash_token(req.refresh_token)
    rt = db.query(RefreshToken).filter(RefreshToken.token_hash == token_hash).first()
    if rt and rt.revoked_at is None:
        rt.revoked_at = datetime.now(timezone.utc)
        db.commit()
    return {"message": "Logged out"}


# --- Phase 2C: self-service usage + admin endpoints ---
@app.get("/api/me/usage", response_model=UsageOut)
def my_usage(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    engine = PolicyEngine(db)
    policy = engine.get_user_policy(current_user.id)
    if not policy:
        raise HTTPException(status_code=404, detail="No policy assigned to this account")
    usage = engine.get_current_usage(current_user.id, policy.tier_scope)
    return UsageOut(
        policy_name=policy.name, tier_scope=policy.tier_scope, period=policy.period,
        action_on_exhaustion=policy.action_on_exhaustion,
        query_threshold=policy.query_threshold, token_threshold=policy.token_threshold,
        queries_used=usage["queries"], tokens_used=usage["tokens"],
    )

@app.get("/api/admin/policies", response_model=List[PolicyOut])
def admin_list_policies(admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    return db.query(Policy).order_by(Policy.id).all()

@app.post("/api/admin/policies", response_model=PolicyOut)
def admin_create_policy(req: PolicyCreate, admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    if db.query(Policy).filter(Policy.name == req.name).first():
        raise HTTPException(status_code=400, detail=f"Policy '{req.name}' already exists")
    p = Policy(**req.model_dump())
    db.add(p)
    db.commit()
    db.refresh(p)
    return p

@app.patch("/api/admin/policies/{policy_id}", response_model=PolicyOut)
def admin_update_policy(policy_id: int, req: PolicyUpdate, admin: User = Depends(require_admin),
                        db: Session = Depends(get_db)):
    p = db.query(Policy).filter(Policy.id == policy_id).first()
    if not p:
        raise HTTPException(status_code=404, detail="Policy not found")
    for field, value in req.model_dump(exclude_unset=True).items():
        setattr(p, field, value)
    db.commit()
    db.refresh(p)
    return p

@app.get("/api/admin/users", response_model=List[UserAdminOut])
def admin_list_users(admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    engine = PolicyEngine(db)
    out = []
    for u in db.query(User).order_by(User.id).all():
        policy = engine.get_user_policy(u.id)
        usage = engine.get_current_usage(u.id, policy.tier_scope if policy else "all")
        out.append(UserAdminOut(
            id=u.id, email=u.email, is_admin=u.is_admin,
            policy_id=policy.id if policy else None, policy_name=policy.name if policy else None,
            queries_used=usage["queries"], tokens_used=usage["tokens"],
        ))
    return out

@app.patch("/api/admin/users/{user_id}/policy")
def admin_reassign_user_policy(user_id: int, req: ReassignPolicyRequest,
                               admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    policy = db.query(Policy).filter(Policy.id == req.policy_id).first()
    if not policy:
        raise HTTPException(status_code=404, detail="Policy not found")
    up = db.query(UserPolicy).filter(UserPolicy.user_id == user_id).first()
    if up:
        up.policy_id = policy.id
    else:
        db.add(UserPolicy(user_id=user_id, policy_id=policy.id))
    db.commit()
    return {"message": f"User {user_id} reassigned to policy '{policy.name}'"}


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
    if tier == LOCAL_RAG:
        return 0.0
    rate = cfg.cost_per_1k_low if tier == LOW else cfg.cost_per_1k_high
    return round(rate * tokens_est / 1000.0, 6)


def _record_usage_safely(user_id: int, tier: str, tokens_est: int, cost: float, downgraded: bool) -> bool:
    """Persist usage in a short-lived session without failing the user response."""
    try:
        with db_module.SessionLocal() as usage_db:
            return bool(PolicyEngine(usage_db).record_usage(
                user_id, tier, tokens_est, cost, downgraded
            ))
    except Exception:
        # Covers failures while opening/closing the session as well as errors
        # before PolicyEngine.record_usage can apply its own rollback handling.
        from gateway.metrics import USAGE_LEDGER_WRITE_FAILURES
        USAGE_LEDGER_WRITE_FAILURES.inc()
        log.exception("Could not open usage-ledger session; continuing without recording usage.")
        return False


def _local_rag_decision(mode: str, retrieval_result) -> dict:
    return {
        "tier": LOCAL_RAG,
        "p_strong": 0.0,
        "threshold": retrieval_result.threshold,
        "confidence": retrieval_result.best_score,
        "mode": mode,
        "source": "local_rag",
        "reasoning": (
            f"3B LOCAL_KB intent + 3C sufficient context "
            f"(rerank={retrieval_result.best_score:.3f} >= {retrieval_result.threshold:.3f})"
        ),
        "signals": [],
    }


def _external_decision(decision: dict, *, reason: str) -> dict:
    out = dict(decision)
    out["reasoning"] = f"{out.get('reasoning') or ''} | {reason}".strip(" |")
    return out

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


@app.get("/api/system/status")
async def system_status():
    """Non-secret feature configuration for the Phase 3F frontend status panel.

    This reports configuration flags, not live reachability of Qdrant, Redis,
    or Ollama. A successful LOCAL-RAG response / semantic-cache hit is the
    runtime proof that those request paths worked.
    """
    return {
        "gateway": "ok",
        "phase_3a": {
            "qdrant_url_configured": bool(cfg.qdrant_url),
            "collection": cfg.qdrant_collection,
        },
        "phase_3b": {"enabled": bool(cfg.enable_mvp3_routing)},
        "phase_3c": {"enabled": bool(cfg.enable_mvp3_retrieval)},
        "phase_3d": {
            "enabled": bool(cfg.enable_mvp3d_rag),
            "model": cfg.local_rag_model,
        },
        "phase_3e": {
            "enabled": bool(cache is not None and cfg.semantic_cache_enabled),
            "threshold": cfg.semantic_cache_threshold,
        },
        "phase_3f": {"enabled": True},
        "observability": {
            "enabled": True,
            "metrics_path": "/metrics",
            "metrics": [
                "routed_to_local_total", "cache_hit_total", "semantic_cache_hits_total",
                "retrieval_latency_ms", "rerank_latency_ms", "reranker_model_load_latency_ms",
                "local_generation_latency_ms", "local_ttft_ms", "external_fallback_total", "estimated_cost_avoided_usd_total",
                "usage_ledger_write_failures_total", "local_rag_accuracy_ratio",
            ],
        },
        "note": "Feature flags are configuration only, not dependency connectivity checks. local_rag_accuracy_ratio is populated by Phase 3G evaluation, not inferred from live traffic.",
    }

@app.post("/api/chat", response_model=ChatResponse, dependencies=[Depends(api_guard)])
async def chat(req: ChatRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    t0 = time.perf_counter()
    request_id = now_id()
    mode = req.mode or cfg.router_mode

    # 1. Policy Evaluation
    policy_engine = PolicyEngine(db)
    policy_decision = policy_engine.evaluate(current_user.id)
    POLICY_DECISION_COUNTER.labels(action=policy_decision.action).inc()
    
    if policy_decision.action == "block":
        raise HTTPException(status_code=402, detail=policy_decision.reason)

    # End the read transaction before slow model inference. Neon may terminate
    # sessions left idle in a transaction while local retrieval/model calls run.
    db.commit()

    # 2. Phase 3E: semantic cache is checked before the 3B intent judge.
    # Policy evaluation remains authoritative. A semantic hit contains the
    # original LOCAL-RAG decision, so 3B/3C/3D can all be skipped.
    augmented_query = req.query
    local_rag_result = None
    local_rag_context = ""
    intent_decision = None
    cache_hit = False
    semantic_hit = None
    tier = LOCAL_RAG
    router_ms = 0.0

    local_cache_allowed = (
        cache is not None
        and req.use_cache
        and cfg.semantic_cache_enabled
        and cfg.enable_mvp3d_rag
        and req.force_tier is None
        and not policy_decision.is_downgraded
    )

    if local_cache_allowed:
        semantic_hit = cache.semantic_get(req.query, mode)
        if semantic_hit and semantic_hit.decision:
            decision = semantic_hit.decision
            cache_hit = True
            CACHE_HIT.labels(cache_type="semantic").inc()
            if decision.get("tier") == LOCAL_RAG:
                ROUTED_TO_LOCAL.inc()
                cached_tokens = max(1, len(semantic_hit.answer) // 4) if semantic_hit.answer else 0
                avoided = cost_of(LOW, cached_tokens)
                if avoided > 0:
                    ESTIMATED_COST_AVOIDED.inc(avoided)
            else:
                avoided = 0.0
            _record_usage_safely(current_user.id, decision.get("tier", LOCAL_RAG), 0, 0.0, False)
            router_ms = 0.0
            total = _log_and_finish(
                request_id, req.query, decision, True, router_ms, 0.0,
                None, t0, 0,
            )
            return ChatResponse(
                request_id=request_id,
                answer=semantic_hit.answer,
                decision=RouteDecisionOut(**{
                    k: decision[k] for k in RouteDecisionOut.model_fields
                }),
                cache_hit=True,
                router_latency_ms=0.0,
                generation_latency_ms=0.0,
                total_latency_ms=round(total, 1),
                tokens_est=0,
                est_cost_usd=0.0,
                downgraded=False,
                ttft_ms=0.0,
                estimated_cost_avoided_usd=avoided if decision.get("tier") == LOCAL_RAG else 0.0,
                sources=getattr(semantic_hit, "sources", []) or [],
            )

    # Cache miss: preserve the existing 3B -> 3C -> 3D path.
    if cfg.enable_mvp3_routing and cfg.enable_mvp3_retrieval:
        intent_decision = judge.classify_intent(req.query)

        if intent_decision.route == "LOCAL_KB":
            try:
                local_rag_result = await asyncio.to_thread(retrieval_pipeline.run, req.query)
                RETRIEVAL_LATENCY.observe(max(0.0, local_rag_result.retrieval_latency_ms))
                RERANK_LATENCY.observe(max(0.0, local_rag_result.latency_ms))
            except Exception:
                # A missing/unloadable cross-encoder or Qdrant client failure must
                # not prevent the existing LOW/HIGH router from answering.
                EXTERNAL_FALLBACK.labels(reason="retrieval_error").inc()
                log.exception("Local retrieval/reranking failed; falling back to external routing")
                local_rag_result = None
            local_rag_context = local_rag_result.context if local_rag_result and local_rag_result.sufficient else ""
            if local_rag_context:
                augmented_query = (
                    "Use the following verified context to answer the question.\n\n"
                    f"Context:\n{local_rag_context}\n\n"
                    f"Question: {req.query}"
                )

    # 3. Routing. A sufficiently grounded 3C result becomes LOCAL-RAG before any
    # external router/model is called. Explicit force_tier remains authoritative.
    use_local_rag = (
        cfg.enable_mvp3d_rag
        and req.force_tier is None
        and not policy_decision.is_downgraded
        and intent_decision is not None
        and intent_decision.route == "LOCAL_KB"
        and local_rag_result is not None
        and local_rag_result.sufficient
        and bool(local_rag_context)
    )
    if use_local_rag:
        decision = _local_rag_decision(mode, local_rag_result)
        router_ms = intent_decision.latency_ms + local_rag_result.latency_ms
        tier = LOCAL_RAG
        ROUTED_TO_LOCAL.inc()
    elif (intent_decision is not None and intent_decision.route == "LOCAL_KB"
          and local_rag_result is None and not policy_decision.is_downgraded):
        decision, router_ms = await get_decision(req.query, mode, req.threshold, req.explain)
        tier = req.force_tier if req.force_tier in (LOW, HIGH) else decision["tier"]
        decision = _external_decision(decision, reason="LOCAL-RAG retrieval/reranking failed; used external fallback")
    elif (intent_decision is not None and intent_decision.route == "LOCAL_KB"
          and local_rag_result is not None and not local_rag_result.sufficient
          and not policy_decision.is_downgraded):
        EXTERNAL_FALLBACK.labels(reason="insufficient_context").inc()
        decision, router_ms = await get_decision(req.query, mode, req.threshold, req.explain)
        tier = req.force_tier if req.force_tier in (LOW, HIGH) else decision["tier"]
        decision = _external_decision(decision, reason="LOCAL-RAG retrieval insufficient; used external fallback")
    elif policy_decision.is_downgraded:
        decision = {"tier": LOW, "p_strong": 0.0, "threshold": 0.0, "confidence": 1.0,
                    "mode": mode, "source": "policy_engine", "reasoning": policy_decision.reason, "signals": []}
        router_ms = 0.0
        tier = LOW
    else:
        decision, router_ms = await get_decision(req.query, mode, req.threshold, req.explain)
        tier = req.force_tier if req.force_tier in (LOW, HIGH) else decision["tier"]

    # 4. Existing exact cache remains in place for non-semantic/external traffic.
    if cache and req.use_cache and tier != LOCAL_RAG:
        hit = cache.get(req.query, mode)
        if hit:
            CACHE_HIT.labels(cache_type="exact").inc()
            decision_for_hit = hit.decision or decision
            _record_usage_safely(
                current_user.id, decision_for_hit.get("tier", tier), 0, 0.0,
                policy_decision.is_downgraded,
            )
            total = _log_and_finish(
                request_id, req.query, decision_for_hit, True, router_ms, 0.0,
                None, t0, 0,
            )
            return ChatResponse(
                request_id=request_id,
                answer=hit.answer,
                decision=RouteDecisionOut(**{
                    k: decision_for_hit[k] for k in RouteDecisionOut.model_fields
                }),
                cache_hit=True,
                router_latency_ms=round(router_ms, 1),
                generation_latency_ms=0.0,
                total_latency_ms=round(total, 1),
                tokens_est=0,
                est_cost_usd=0.0,
                downgraded=policy_decision.is_downgraded,
                sources=getattr(hit, "sources", []) or [],
            )

    if tier == LOCAL_RAG:
        stats = await local_rag_client.generate(
            RagRequest(req.query, local_rag_context),
            max_tokens=req.max_tokens or cfg.max_tokens,
            temperature=cfg.local_rag_temperature,
        )
        if stats.error:
            # Capture failed local generation latency before replacing stats with
            # the external fallback result.
            LOCAL_GENERATION_LATENCY.observe(max(0.0, stats.latency_ms))
            if stats.ttft_ms is not None:
                LOCAL_TTFT.observe(max(0.0, stats.ttft_ms))
            EXTERNAL_FALLBACK.labels(reason="generation_error").inc()
            # Local generation is an optimization, not a hard dependency. Fall back
            # to the existing router/external path without changing LOW/HIGH semantics.
            fallback_decision, fallback_router_ms = await get_decision(req.query, mode, req.threshold, req.explain)
            tier = req.force_tier if req.force_tier in (LOW, HIGH) else fallback_decision["tier"]
            decision = _external_decision(fallback_decision, reason="LOCAL-RAG generation failed; used external fallback")
            router_ms += fallback_router_ms
            client = tier_client(tier)
            messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": augmented_query}]
            stats = await collect_stream(client.stream_chat(messages, max_tokens=req.max_tokens or cfg.max_tokens))
    else:
        client = tier_client(tier)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": augmented_query}]
        stats = await collect_stream(client.stream_chat(messages, max_tokens=req.max_tokens or cfg.max_tokens))

    if stats.error:
        _log_and_finish(request_id, req.query, decision, False, router_ms, stats.latency_ms, None, t0, 0, stats.error)
        raise HTTPException(status_code=502, detail=f"{tier} tier error: {stats.error}")

    # 5. Record Usage and Return. Use a fresh DB session after inference so the
    # request's authentication/policy session never sits idle during model work.
    cost = cost_of(tier, stats.tokens_est)
    _record_usage_safely(
        current_user.id, tier, stats.tokens_est, cost, policy_decision.is_downgraded
    )

    estimated_avoided = 0.0
    if tier == LOCAL_RAG:
        LOCAL_GENERATION_LATENCY.observe(max(0.0, stats.latency_ms))
        if stats.ttft_ms is not None:
            LOCAL_TTFT.observe(max(0.0, stats.ttft_ms))
        estimated_avoided = cost_of(LOW, stats.tokens_est)
        if estimated_avoided > 0:
            ESTIMATED_COST_AVOIDED.inc(estimated_avoided)

    if cache and req.use_cache:
        cache.put(
            req.query, mode, stats.text, tier, decision["p_strong"], decision=decision,
            sources=local_rag_result.sources if tier == LOCAL_RAG and local_rag_result is not None else [],
        )
        
    total = _log_and_finish(request_id, req.query, decision, False, router_ms, stats.latency_ms, stats.ttft_ms, t0, stats.tokens_est)
    return ChatResponse(
        request_id=request_id, answer=stats.text,
        decision=RouteDecisionOut(**{k: decision.get(k, 0) for k in RouteDecisionOut.model_fields}),
        cache_hit=False, router_latency_ms=round(router_ms, 1), generation_latency_ms=round(stats.latency_ms, 1),
        total_latency_ms=round(total, 1), tokens_est=stats.tokens_est, est_cost_usd=cost,
        downgraded=policy_decision.is_downgraded,
        ttft_ms=round(stats.ttft_ms, 1) if stats.ttft_ms is not None else None,
        estimated_cost_avoided_usd=estimated_avoided,
        sources=local_rag_result.sources if tier == LOCAL_RAG and local_rag_result is not None else [],
    )

@app.post("/api/chat/stream", dependencies=[Depends(api_guard)])
async def chat_stream(req: ChatRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    t0 = time.perf_counter()
    request_id = now_id()
    mode = req.mode or cfg.router_mode

    # 1. Policy Evaluation
    policy_engine = PolicyEngine(db)
    policy_decision = policy_engine.evaluate(current_user.id)
    POLICY_DECISION_COUNTER.labels(action=policy_decision.action).inc()
    if policy_decision.action == "block":
        raise HTTPException(status_code=402, detail=policy_decision.reason)

    # Do not leave a Neon transaction open during retrieval or model streaming.
    db.commit()

    # 2. Phase 3E: semantic cache is checked before the 3B intent judge.
    # Policy evaluation remains authoritative. A semantic hit skips 3B/3C/3D.
    augmented_query = req.query
    local_rag_result = None
    local_rag_context = ""
    intent_decision = None
    cache_hit = False
    semantic_hit = None
    tier = LOCAL_RAG
    router_ms = 0.0

    local_cache_allowed = (
        cache is not None
        and req.use_cache
        and cfg.semantic_cache_enabled
        and cfg.enable_mvp3d_rag
        and req.force_tier is None
        and not policy_decision.is_downgraded
    )

    if local_cache_allowed:
        semantic_hit = cache.semantic_get(req.query, mode)
        if semantic_hit and semantic_hit.decision:
            decision = semantic_hit.decision
            cache_hit = True
            CACHE_HIT.labels(cache_type="semantic").inc()
            if decision.get("tier") == LOCAL_RAG:
                ROUTED_TO_LOCAL.inc()
            _record_usage_safely(current_user.id, decision.get("tier", LOCAL_RAG), 0, 0.0, False)
            router_ms = 0.0

    # Cache miss: preserve the existing 3B -> 3C -> 3D path.
    if not cache_hit and cfg.enable_mvp3_routing and cfg.enable_mvp3_retrieval:
        intent_decision = judge.classify_intent(req.query)

        if intent_decision.route == "LOCAL_KB":
            try:
                local_rag_result = await asyncio.to_thread(retrieval_pipeline.run, req.query)
                RETRIEVAL_LATENCY.observe(max(0.0, local_rag_result.retrieval_latency_ms))
                RERANK_LATENCY.observe(max(0.0, local_rag_result.latency_ms))
            except Exception:
                # A missing/unloadable cross-encoder or Qdrant client failure must
                # not prevent the existing LOW/HIGH router from answering.
                EXTERNAL_FALLBACK.labels(reason="retrieval_error").inc()
                log.exception("Local retrieval/reranking failed; falling back to external routing")
                local_rag_result = None
            local_rag_context = local_rag_result.context if local_rag_result and local_rag_result.sufficient else ""
            if local_rag_context:
                augmented_query = (
                    "Use the following verified context to answer the question.\n\n"
                    f"Context:\n{local_rag_context}\n\n"
                    f"Question: {req.query}"
                )

    # 3. A sufficiently grounded 3C result becomes LOCAL-RAG before any external
    # router/model is called. Explicit force_tier and policy downgrade remain authoritative.
    if not cache_hit:
        use_local_rag = (
            cfg.enable_mvp3d_rag
            and req.force_tier is None
            and not policy_decision.is_downgraded
            and intent_decision is not None
            and intent_decision.route == "LOCAL_KB"
            and local_rag_result is not None
            and local_rag_result.sufficient
            and bool(local_rag_context)
        )
        if use_local_rag:
            decision = _local_rag_decision(mode, local_rag_result)
            router_ms = intent_decision.latency_ms + local_rag_result.latency_ms
            tier = LOCAL_RAG
            ROUTED_TO_LOCAL.inc()
        elif (intent_decision is not None and intent_decision.route == "LOCAL_KB"
              and local_rag_result is None and not policy_decision.is_downgraded):
            decision, router_ms = await get_decision(req.query, mode, req.threshold, req.explain)
            tier = req.force_tier if req.force_tier in (LOW, HIGH) else decision["tier"]
            decision = _external_decision(decision, reason="LOCAL-RAG retrieval/reranking failed; used external fallback")
        elif (intent_decision is not None and intent_decision.route == "LOCAL_KB"
              and local_rag_result is not None and not local_rag_result.sufficient
              and not policy_decision.is_downgraded):
            EXTERNAL_FALLBACK.labels(reason="insufficient_context").inc()
            decision, router_ms = await get_decision(req.query, mode, req.threshold, req.explain)
            tier = req.force_tier if req.force_tier in (LOW, HIGH) else decision["tier"]
            decision = _external_decision(decision, reason="LOCAL-RAG retrieval insufficient; used external fallback")
        elif policy_decision.is_downgraded:
            decision = {
                "tier": LOW, "p_strong": 0.0, "threshold": 0.0, "confidence": 1.0,
                "mode": mode, "source": "policy_engine", "reasoning": policy_decision.reason, "signals": [],
            }
            router_ms = 0.0
            tier = LOW
        else:
            decision, router_ms = await get_decision(req.query, mode, req.threshold, req.explain)
            tier = req.force_tier if req.force_tier in (LOW, HIGH) else decision["tier"]

        # Existing exact cache is retained for LOW/HIGH traffic.
        if cache and req.use_cache and tier != LOCAL_RAG:
            hit = cache.get(req.query, mode)
            if hit:
                CACHE_HIT.labels(cache_type="exact").inc()
                decision = hit.decision or decision
                _record_usage_safely(
                    current_user.id, decision.get("tier", tier), 0, 0.0,
                    policy_decision.is_downgraded,
                )
                cache_hit = True
                # Exact cache hit uses the same streaming fast path below.
                semantic_hit = hit

    async def gen() -> AsyncIterator[bytes]:
        nonlocal decision, tier, router_ms

        def sse(event: str, data: dict) -> bytes:
            return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()

        yield sse("meta", {
            "request_id": request_id,
            "decision": decision,
            "cache_hit": cache_hit,
            "router_latency_ms": round(router_ms, 1),
            "downgraded": policy_decision.is_downgraded,
        })

        text_parts: list[str] = []
        if cache_hit and semantic_hit is not None:
            cached_text = semantic_hit.answer
            if cached_text:
                text_parts.append(cached_text)
                yield sse("delta", {"text": cached_text})
            total = _log_and_finish(
                request_id, req.query, decision, True, router_ms, 0.0,
                0.0, t0, 0,
            )
            cached_tokens = max(1, len(cached_text) // 4) if cached_text else 0
            avoided = cost_of(LOW, cached_tokens) if decision.get("tier") == LOCAL_RAG else 0.0
            if avoided > 0:
                ESTIMATED_COST_AVOIDED.inc(avoided)
            yield sse("done", {
                "total_latency_ms": round(total, 1),
                "cache_hit": True,
                "tokens_est": 0,
                "est_cost_usd": 0.0,
                "estimated_cost_avoided_usd": avoided,
                "ttft_ms": 0.0,
                "tier": tier,
            })
            return

        ttft_ms = None
        gen_t0 = time.perf_counter()
        err = None

        if tier == LOCAL_RAG:
            stream_client = local_rag_client.stream(
                RagRequest(req.query, local_rag_context),
                max_tokens=req.max_tokens or cfg.max_tokens,
                temperature=cfg.local_rag_temperature,
            )
        else:
            client = tier_client(tier)
            messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": augmented_query}]
            stream_client = client.stream_chat(messages, max_tokens=req.max_tokens or cfg.max_tokens)

        async for chunk in stream_client:
            if chunk.error:
                err = chunk.error
                if tier == LOCAL_RAG:
                    # Local RAG is an optimization, not a hard dependency.
                    LOCAL_GENERATION_LATENCY.observe(max(0.0, (time.perf_counter() - gen_t0) * 1000))
                    if ttft_ms is not None:
                        LOCAL_TTFT.observe(max(0.0, ttft_ms))
                    EXTERNAL_FALLBACK.labels(reason="generation_error").inc()
                    fallback_decision, fallback_router_ms = await get_decision(
                        req.query, mode, req.threshold, req.explain
                    )
                    fallback_tier = req.force_tier if req.force_tier in (LOW, HIGH) else fallback_decision["tier"]
                    decision = _external_decision(
                        fallback_decision,
                        reason="LOCAL-RAG unavailable; used external fallback",
                    )
                    tier = fallback_tier
                    router_ms += fallback_router_ms
                    yield sse("fallback", {"from": LOCAL_RAG, "to": tier, "reason": err})

                    text_parts = []
                    ttft_ms = None
                    gen_t0 = time.perf_counter()
                    err = None
                    client = tier_client(tier)
                    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": augmented_query}]
                    async for fallback_chunk in client.stream_chat(
                        messages, max_tokens=req.max_tokens or cfg.max_tokens
                    ):
                        if fallback_chunk.error:
                            err = fallback_chunk.error
                            yield sse("error", {"message": err})
                            break
                        if fallback_chunk.delta:
                            if ttft_ms is None:
                                ttft_ms = (time.perf_counter() - gen_t0) * 1000
                            text_parts.append(fallback_chunk.delta)
                            yield sse("delta", {"text": fallback_chunk.delta})
                        if fallback_chunk.done:
                            break
                    break

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

        estimated_avoided = 0.0
        if not err:
            _record_usage_safely(
                current_user.id, tier, tokens_est, cost, policy_decision.is_downgraded
            )
            if tier == LOCAL_RAG:
                LOCAL_GENERATION_LATENCY.observe(max(0.0, gen_ms))
                if ttft_ms is not None:
                    LOCAL_TTFT.observe(max(0.0, ttft_ms))
                estimated_avoided = cost_of(LOW, tokens_est)
                if estimated_avoided > 0:
                    ESTIMATED_COST_AVOIDED.inc(estimated_avoided)
            if cache and req.use_cache:
                cache.put(
                    req.query, mode, full_text, tier, decision["p_strong"], decision=decision,
                    sources=local_rag_result.sources if tier == LOCAL_RAG and local_rag_result is not None else [],
                )

        total = _log_and_finish(
            request_id, req.query, decision, False, router_ms, gen_ms,
            ttft_ms, t0, tokens_est, err,
        )
        yield sse("done", {
            "total_latency_ms": round(total, 1),
            "cache_hit": cache_hit,
            "tokens_est": tokens_est,
            "est_cost_usd": cost,
            "estimated_cost_avoided_usd": estimated_avoided,
            "ttft_ms": round(ttft_ms, 1) if ttft_ms is not None else None,
            "tier": tier,
        })

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
