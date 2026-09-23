"""Turn RouteLLM's GPT-4-judge scores into routing labels.

routellm/gpt4_dataset has, for every Arena-style prompt, a `mixtral_score`
(1-5): how good a *weak* model's (Mixtral-8x7B) answer was, judged by GPT-4
against GPT-4's own answer.

    5 -> weak model matched the strong one   -> "simple"
    4 -> weak model was close                -> "medium"
    <=3 -> weak model was noticeably worse   -> "complex"

Binary routing label (what the router must predict):
    score >= threshold -> "low"  (cheap tier is good enough)
    otherwise          -> "high" (needs the strong tier)

This is a *proxy* for complexity: it measures "did a weaker model struggle",
which is exactly what a router needs to know.
"""
from __future__ import annotations

LOW = "low"
HIGH = "high"
COMPLEXITY_LEVELS = ("simple", "medium", "complex")


def score_to_complexity(score: int) -> str:
    if score >= 5:
        return "simple"
    if score == 4:
        return "medium"
    return "complex"


def score_to_tier(score: int, threshold: int = 4) -> str:
    return LOW if score >= threshold else HIGH
