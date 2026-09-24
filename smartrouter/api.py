"""Phase 2 router service exposing POST /route."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .classifier_router import ClassifierRouter, RulesRouter
from .config import load_settings

log = logging.getLogger("smartrouter.api")
cfg = load_settings()
state: dict = {"router": None, "source": "unloaded"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        state["router"] = ClassifierRouter.load(cfg.router_artifact_path, min_confidence=cfg.router_min_confidence)
        state["source"] = "classifier"
        log.info("Loaded classifier artifact from %s", cfg.router_artifact_path)
    except Exception as e:
        state["router"] = RulesRouter()
        state["source"] = "rules_fallback"
        log.warning("Could not load classifier artifact (%s: %s). Serving with static rules fallback.", type(e).__name__, e)
    yield
    state.clear()


app = FastAPI(title="LLM Smart Router - Routing Service", version="0.2.0", lifespan=lifespan)


class RouteRequest(BaseModel):
    query: str = Field(..., min_length=0, description="The user prompt to classify.")
    mode: Optional[str] = Field(None, description="Cost/quality preset, e.g. 'aggressive'|'balanced'|'conservative'.")
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Override P(strong) threshold directly.")
    explain: bool = Field(False, description="Include human-readable reasoning string.")


class RouteResponse(BaseModel):
    tier: str
    confidence: float
    p_strong: float
    threshold: float
    mode: str
    source: str
    latency_ms: float
    reasoning: Optional[str] = None
    signals: list[str] = []


@app.post("/route", response_model=RouteResponse)
def route(req: RouteRequest) -> RouteResponse:
    router = state["router"]
    if router is None:
        raise HTTPException(status_code=503, detail="Router not ready")
    try:
        mode = req.mode or (cfg.default_route_mode if req.threshold is None else None)
        d = router.route(req.query, mode=mode, threshold=req.threshold, explain=req.explain)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return RouteResponse(
        tier=d.tier, confidence=d.confidence, p_strong=d.p_strong, threshold=d.threshold,
        mode=d.mode, source=d.source, latency_ms=d.latency_ms, reasoning=d.reasoning, signals=d.signals,
    )


@app.get("/health")
def health() -> dict:
    router = state["router"]
    return {
        "status": "ok" if router is not None else "starting",
        "source": state["source"],
        "modes": sorted(getattr(router, "modes", {})) if router is not None else [],
        "artifact_path": str(cfg.router_artifact_path),
    }


@app.get("/modes")
def modes() -> dict:
    router = state["router"]
    if router is None or not getattr(router, "modes", None):
        return {"modes": {}, "default": cfg.default_route_mode}
    return {"modes": {k: round(v, 4) for k, v in router.modes.items()}, "default": router.default_mode}
