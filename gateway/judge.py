import time
from typing import List
from pydantic import BaseModel
from gateway.settings import load_gateway_settings

class JudgeDecision(BaseModel):
    route: str
    intent_score: float
    matched_signals: List[str]
    latency_ms: float

class QueryJudge:
    def __init__(self):
        # Load the configuration using the gateway's settings manager
        cfg = load_gateway_settings()
        self.keywords = [kw.lower() for kw in cfg.indian_context_keywords]
        self.threshold = cfg.intent_threshold

    def classify_intent(self, query: str) -> JudgeDecision:
        start_time = time.perf_counter()
        
        if not query or not query.strip():
            latency = (time.perf_counter() - start_time) * 1000
            return JudgeDecision(
                route="EXTERNAL_LLM",
                intent_score=0.0,
                matched_signals=[],
                latency_ms=latency
            )
            
        query_lower = query.lower()
        matched = [kw for kw in self.keywords if kw in query_lower]
        
        # Heuristic scoring based on keyword matches for MVP 3B
        score = 0.0
        if matched:
            # Base score of 0.75 guarantees routing to LOCAL_KB if any keyword matches
            score = 0.75 + (0.05 * len(matched)) 
            score = min(score, 1.0)
            
        route = "LOCAL_KB" if score > self.threshold else "EXTERNAL_LLM"
        
        latency = (time.perf_counter() - start_time) * 1000
        
        return JudgeDecision(
            route=route,
            intent_score=score,
            matched_signals=matched,
            latency_ms=latency
        )
