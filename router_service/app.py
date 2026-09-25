"""Router microservice — Phase 2's API contract, run as its own FastAPI app.

    POST /route  -> { query, tier: "low"|"high", confidence, reasoning_optional }
    GET  /health -> router status (ok | degraded) + calibrated thresholds
    GET  /modes  -> available cost-quality presets

Stateless: holds only the loaded model in memory, so it scales horizontally
behind a load balancer independently of the gateway, as the plan requires.

Run:
    uvicorn router_service.app:app --host 0.0.0.0 --port 8001
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from gateway.router_loader import load_router  # noqa: E402
from gateway.settings import load_gateway_settings  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("router_service")

cfg = load_gateway_settings()
router, router_status = load_router(cfg.router_artifact_path)
log.info("Router status: %s", router_status)

app = FastAPI(title="LLM Smart Router - Router Service", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=cfg.cors_origins, allow_methods=["*"], allow_headers=["*"])


class RouteRequest(BaseModel):
    query: str = Field(..., description="The user prompt to classify.")
    mode: Optional[str] = Field(None, description="Cost-quality preset: aggressive|balanced|conservative.")
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Override: custom P(strong) cut-off.")
    explain: bool = Field(False, description="Include a human-readable `reasoning` string.")


class RouteResponse(BaseModel):
    query_preview: str
    tier: str
    confidence: float
    p_strong: float
    threshold: float
    mode: str
    source: str
    latency_ms: float
    reasoning: Optional[str] = None
    signals: list[str] = []


@app.get("/health")
def health():
    return {"status": router_status["status"], "router": router_status}


@app.get("/modes")
def modes():
    return {"default_mode": getattr(router, "default_mode", None), "modes": getattr(router, "modes", {})}


@app.post("/route", response_model=RouteResponse)
def route(req: RouteRequest):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=422, detail="`query` must not be empty")
    try:
        d = router.route(req.query, mode=req.mode, threshold=req.threshold, explain=req.explain)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return RouteResponse(
        query_preview=(req.query[:80] + ("..." if len(req.query) > 80 else "")),
        tier=d.tier, confidence=d.confidence, p_strong=d.p_strong, threshold=d.threshold,
        mode=d.mode, source=d.source, latency_ms=d.latency_ms, reasoning=d.reasoning,
        signals=d.signals,
    )
