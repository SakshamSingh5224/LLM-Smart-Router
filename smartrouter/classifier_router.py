"""ClassifierRouter runtime implementation for model loading and scoring."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Dict, Any, Union
import joblib

from .rules import RouteDecision, RulesRouter
from .features import extract_signals


class ClassifierRouter:
    """Production router serving predictions from a trained joblib artifact."""

    def __init__(self, artifact: Dict[str, Any], min_confidence: float = 0.0):
        self.pipeline = artifact["pipeline"]
        self.model = artifact["model"]
        self.modes = artifact["thresholds"]
        self.default_mode = "balanced"
        self.min_confidence = min_confidence
        self.fallback_router = RulesRouter()

    @classmethod
    def load(cls, path: Union[str, Path], min_confidence: float = 0.0) -> ClassifierRouter:
        artifact = joblib.load(path)
        return cls(artifact, min_confidence=min_confidence)

    def predict_proba(self, queries: list[str]) -> list[float]:
        feats = self.pipeline.transform(queries)
        return self.model.predict_proba(feats)[:, 1].tolist()

    def route(self, query: str, mode: Optional[str] = None, threshold: Optional[float] = None, explain: bool = False) -> RouteDecision:
        t0 = time.perf_counter()
        active_mode = mode or self.default_mode

        if threshold is not None:
            eff_threshold = float(threshold)
        elif active_mode in self.modes:
            eff_threshold = self.modes[active_mode]
        else:
            raise ValueError(f"Unknown mode '{active_mode}'. Available modes: {list(self.modes.keys())}")

        try:
            p_strong = self.predict_proba([query])[0]
            confidence = round(abs(p_strong - eff_threshold) * 2, 4)

            if confidence < self.min_confidence:
                return self.fallback_router.route(query, mode=mode, threshold=threshold, explain=explain)

            tier = "high" if p_strong >= eff_threshold else "low"
            latency_ms = (time.perf_counter() - t0) * 1000.0

            sig = extract_signals(query)
            signals_list = [k for k, v in sig.items() if v and k not in ("word_count", "char_count")]

            reasoning = None
            if explain:
                reasoning = f"Classifier: p_strong={p_strong:.4f}, threshold={eff_threshold:.4f}, mode={active_mode}"

            return RouteDecision(
                tier=tier,
                confidence=confidence,
                p_strong=round(p_strong, 4),
                threshold=eff_threshold,
                mode=active_mode,
                source="classifier",
                latency_ms=round(latency_ms, 3),
                reasoning=reasoning,
                signals=signals_list,
            )
        except Exception:
            return self.fallback_router.route(query, mode=mode, threshold=threshold, explain=explain)
