"""Static rule-based router: the safety net used when the classifier is unavailable.

Deterministic, dependency-free and explainable. Not meant to be accurate - just
sensible (code / math / long / constrained prompts -> HIGH, everything else -> LOW).
"""
from __future__ import annotations

from dataclasses import dataclass

from .features import CODE_RE, FORMAT_RE, GREETING_RE, MATH_RE, MAX_FEATURE_CHARS, REASON_RE, describe_signals

RULE_THRESHOLD = 0.5


@dataclass
class RuleDecision:
    tier: str
    score: float          # pseudo-probability that the strong tier is needed, in [0, 1]
    signals: list


def rule_score(text: str) -> float:
    t = (text or "")[:MAX_FEATURE_CHARS]
    words = len(t.split())
    s = 0.25
    s += 0.30 * bool(CODE_RE.search(t))
    s += 0.25 * bool(MATH_RE.search(t))
    s += 0.15 * bool(REASON_RE.search(t))
    s += 0.10 * bool(FORMAT_RE.search(t))
    s += 0.15 * (words > 150)
    s += 0.10 * (words > 400)
    s -= 0.15 * bool(GREETING_RE.search(t))
    s -= 0.10 * (words < 8)
    return float(min(1.0, max(0.0, s)))


def rule_route(text: str) -> RuleDecision:
    score = rule_score(text)
    return RuleDecision("high" if score >= RULE_THRESHOLD else "low", score, describe_signals(text))
