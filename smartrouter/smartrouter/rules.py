"""Static heuristic fallback router used when the classifier is unavailable."""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import List, Optional
from .features import extract_signals

@dataclass
class RouteDecision:
    tier: str
    confidence: float
    p_strong: float
    threshold: float
    mode: str
    source: str
    latency_ms: float
    reasoning: Optional[str] = None
    signals: List[str] = field(default_factory=list)

class RulesRouter:
    def __init__(self, default_mode: str = "balanced"):
        self.default_mode = default_mode
        self.modes = {"aggressive": 0.70, "balanced": 0.50, "conservative": 0.30}

    def route(self, query: str, mode: Optional[str] = None, threshold: Optional[float] = None, explain: bool = False) -> RouteDecision:
        t0 = time.perf_counter()
        active_mode = mode or self.default_mode
        eff_threshold = threshold if threshold is not None else self.modes.get(active_mode, 0.50)

        sig = extract_signals(query)
        detected_signals = []
        score = 0.20
        if sig["has_code"]: score += 0.35; detected_signals.append("contains_code")
        if sig["has_math"]: score += 0.25; detected_signals.append("contains_math")
        if sig["has_reasoning"]: score += 0.20; detected_signals.append("complex_reasoning")
        if sig["is_long"]: score += 0.15; detected_signals.append("long_prompt")

        p_strong = min(1.0, score)
        tier = "high" if p_strong >= eff_threshold else "low"
        
        return RouteDecision(
            tier=tier,
            confidence=round(abs(p_strong - eff_threshold) * 2, 4),
            p_strong=round(p_strong, 4),
            threshold=eff_threshold,
            mode=active_mode,
            source="rules_fallback",
            latency_ms=round((time.perf_counter() - t0) * 1000.0, 3),
            reasoning=f"Rule-based fallback: score={p_strong:.2f}, threshold={eff_threshold:.2f}" if explain else None,
            signals=detected_signals,
        )
