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
        log.info("Loaded classifier artifact")
    except Exception as e:
        state["router"] = RulesRouter()
        state["source"] = "rules_fallback"
        log.warning("Serving with static rules fallback. Run train_classifier.py to enable classifier.")
    yield
    state.clear()

app = FastAPI(title="LLM Smart Router", lifespan=lifespan)

class RouteRequest(BaseModel):
    query: str = Field(...)
    mode: Optional[str] = None
    threshold: Optional[float] = None
    explain: bool = False

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
    if state["router"] is None: raise HTTPException(status_code=503, detail="Router not ready")
    try:
        d = state["router"].route(req.query, mode=req.mode or (cfg.default_route_mode if req.threshold is None else None), threshold=req.threshold, explain=req.explain)
        return RouteResponse(**d.__dict__)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

@app.get("/health")
def health() -> dict:
    return {"status": "ok" if state["router"] else "starting", "source": state["source"]}
