"""ClassifierRouter runtime implementation for model loading and scoring."""
from __future__ import annotations
import time
from pathlib import Path
from typing import Optional, Dict, Any, Union
import joblib
from .rules import RouteDecision, RulesRouter
from .features import extract_signals

class ClassifierRouter:
    def __init__(self, artifact: Dict[str, Any], min_confidence: float = 0.0):
        self.pipeline, self.model = artifact["pipeline"], artifact["model"]
        self.modes, self.default_mode, self.min_confidence = artifact["thresholds"], "balanced", min_confidence
        self.fallback_router = RulesRouter()

    @classmethod
    def load(cls, path: Union[str, Path], min_confidence: float = 0.0) -> ClassifierRouter:
        return cls(joblib.load(path), min_confidence=min_confidence)

    def predict_proba(self, queries: list[str]) -> list[float]:
        return self.model.predict_proba(self.pipeline.transform(queries))[:, 1].tolist()

    def route(self, query: str, mode: Optional[str] = None, threshold: Optional[float] = None, explain: bool = False) -> RouteDecision:
        t0 = time.perf_counter()
        active_mode = mode or self.default_mode
        eff_threshold = float(threshold) if threshold is not None else self.modes.get(active_mode)
        
        if eff_threshold is None:
            raise ValueError(f"Unknown mode '{active_mode}'")

        try:
            p_strong = self.predict_proba([query])[0]
            confidence = round(abs(p_strong - eff_threshold) * 2, 4)

            if confidence < self.min_confidence:
                return self.fallback_router.route(query, mode=mode, threshold=threshold, explain=explain)

            tier = "high" if p_strong >= eff_threshold else "low"
            sig = extract_signals(query)
            
            return RouteDecision(
                tier=tier, confidence=confidence, p_strong=round(p_strong, 4), threshold=eff_threshold,
                mode=active_mode, source="classifier", latency_ms=round((time.perf_counter() - t0) * 1000.0, 3),
                reasoning=f"Classifier: p_strong={p_strong:.4f}, threshold={eff_threshold:.4f}" if explain else None,
                signals=[k for k, v in sig.items() if v and k not in ("word_count", "char_count")],
            )
        except Exception:
            return self.fallback_router.route(query, mode=mode, threshold=threshold, explain=explain)
