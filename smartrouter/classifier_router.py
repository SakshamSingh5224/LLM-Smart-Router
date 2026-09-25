"""Runtime router: prompt -> tier decision.

* ClassifierRouter : trained classifier + calibrated thresholds ("cost-quality slider")
* RulesRouter      : static fallback used when no model artifact is available
* CircuitBreaker   : after repeated classifier failures, go straight to the rules for a while
Any classifier error is absorbed: the caller always gets a decision.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .features import describe_signals
from .rules import RULE_THRESHOLD, rule_route

LOW, HIGH = "low", "high"


@dataclass
class Decision:
    tier: str
    p_strong: float
    threshold: float
    confidence: float
    mode: str
    source: str                      # "classifier" | "rules_fallback"
    latency_ms: float = 0.0
    reasoning: Optional[str] = None
    signals: list = field(default_factory=list)


def _confidence(p: float, thr: float, tier: str) -> float:
    """Normalised distance from the decision boundary, in [0, 1]."""
    if tier == HIGH:
        c = (p - thr) / max(1.0 - thr, 1e-9)
    else:
        c = (thr - p) / max(thr, 1e-9)
    return float(min(1.0, max(0.0, c)))


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, reset_after_s: float = 30.0):
        self.failure_threshold, self.reset_after_s = failure_threshold, reset_after_s
        self.failures, self.opened_at = 0, None

    def allow(self) -> bool:
        if self.opened_at is None:
            return True
        if time.monotonic() - self.opened_at >= self.reset_after_s:  # half-open: try again
            self.opened_at, self.failures = None, self.failure_threshold - 1
            return True
        return False

    def success(self) -> None:
        self.failures, self.opened_at = 0, None

    def failure(self) -> None:
        self.failures += 1
        if self.failures >= self.failure_threshold:
            self.opened_at = time.monotonic()

    @property
    def is_open(self) -> bool:
        return self.opened_at is not None


class RulesRouter:
    """Used when the classifier artifact cannot be loaded (degraded mode)."""

    source = "rules_fallback"
    modes: dict = {}

    def route(self, query: str, mode: Optional[str] = None, threshold: Optional[float] = None,
              explain: bool = False) -> Decision:
        t0 = time.perf_counter()
        rd = rule_route(query if isinstance(query, str) else str(query))
        return Decision(rd.tier, rd.score, RULE_THRESHOLD, _confidence(rd.score, RULE_THRESHOLD, rd.tier),
                        "fallback", self.source, (time.perf_counter() - t0) * 1000,
                        ("static rules: " + (", ".join(rd.signals) or "no strong signals")) if explain else None,
                        rd.signals)


class ClassifierRouter:
    source = "classifier"

    def __init__(self, artifact: dict, min_confidence: float = 0.0, max_chars: int = 20000,
                 breaker: Optional[CircuitBreaker] = None):
        self.artifact = artifact
        self.featurizer = artifact["featurizer"]
        self.clf = artifact["clf"]
        self.modes: dict = dict(artifact["thresholds"])
        self.default_mode: str = artifact.get("default_mode", "balanced")
        self.min_confidence, self.max_chars = min_confidence, max_chars
        self.breaker = breaker or CircuitBreaker()

    @classmethod
    def load(cls, path, **kw) -> "ClassifierRouter":
        import joblib

        artifact = joblib.load(Path(path))
        if artifact.get("schema") != 1:
            raise ValueError(f"Unsupported artifact schema: {artifact.get('schema')}")
        return cls(artifact, **kw)

    def predict_proba(self, texts: list) -> np.ndarray:
        x = self.featurizer.transform(texts)
        return self.clf.predict_proba(x)[:, 1]

    def route(self, query: str, mode: Optional[str] = None, threshold: Optional[float] = None,
              explain: bool = False) -> Decision:
        t0 = time.perf_counter()
        mode = mode or self.default_mode
        if threshold is None and mode not in self.modes:
            raise ValueError(f"unknown mode '{mode}'. Available: {sorted(self.modes)}")
        q = (query if isinstance(query, str) else str(query))[: self.max_chars]

        try:
            if not self.breaker.allow():
                raise RuntimeError("circuit open")
            p = float(self.predict_proba([q])[0])
            self.breaker.success()
        except Exception:  # noqa: BLE001  - never let a model problem block a request
            self.breaker.failure()
            d = RulesRouter().route(q, explain=explain)
            d.latency_ms = (time.perf_counter() - t0) * 1000
            return d

        thr = float(threshold) if threshold is not None else self.modes[mode]
        tier = HIGH if p > thr else LOW
        conf = _confidence(p, thr, tier)
        signals = describe_signals(q)
        note = ""
        if tier == LOW and conf < self.min_confidence:  # low-confidence -> safer (strong) tier
            tier, note = HIGH, f"; escalated: confidence {conf:.2f} < {self.min_confidence:.2f}"
        reasoning = None
        if explain:
            reasoning = (f"P(strong needed)={p:.3f} {'>' if p > thr else '<='} threshold {thr:.3f} "
                         f"(mode={'custom' if threshold is not None else mode}); "
                         f"signals: {', '.join(signals) or 'none'}{note}")
        return Decision(tier, p, thr, conf, "custom" if threshold is not None else mode, self.source,
                        (time.perf_counter() - t0) * 1000, reasoning, signals)
