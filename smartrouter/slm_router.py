"""Prompt-based routing with a small local LLM (pure logic, no network).

The SLM is asked for a tiny JSON object. Anything we cannot parse falls back to
the HIGH tier - the safe default required by the project plan.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from .labels import COMPLEXITY_LEVELS, HIGH, LOW

MAX_QUERY_CHARS = 1200  # keep router prompts short -> lower latency

SYSTEM_PROMPT = """You are a query-complexity classifier inside an LLM routing gateway.
Decide how capable a model must be to answer the user's query well.

simple  : greetings, chit-chat, short factual lookups, rewording, translation of short text,
          yes/no answers, basic how-to advice. A tiny model answers these fine.
medium  : summaries, explanations, short lists, simple code snippets, moderate writing.
complex : multi-step reasoning, math, non-trivial coding/debugging, long or technical writing,
          strict output formats, roleplay with many constraints, expert domain analysis.

Reply with JSON only: {"complexity": "simple|medium|complex", "confidence": <0.0-1.0>}"""

# JSON schema handed to Ollama's structured-output `format` option.
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "complexity": {"type": "string", "enum": list(COMPLEXITY_LEVELS)},
        "confidence": {"type": "number"},
    },
    "required": ["complexity", "confidence"],
}


@dataclass
class RouteDecision:
    tier: str
    complexity: str
    confidence: float
    parse_ok: bool
    latency_ms: float = 0.0
    raw: str = ""


def build_messages(query: str) -> list[dict]:
    q = query if len(query) <= MAX_QUERY_CHARS else query[:MAX_QUERY_CHARS] + " ...[truncated]"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Query:\n\"\"\"\n{q}\n\"\"\""},
    ]


def complexity_to_tier(complexity: str, medium_tier: str = LOW) -> str:
    if complexity == "simple":
        return LOW
    if complexity == "complex":
        return HIGH
    return medium_tier


def parse_decision(text: str, medium_tier: str = LOW) -> RouteDecision:
    """Parse the SLM output; on any problem return a safe HIGH decision."""
    obj: Optional[dict] = None
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        m = re.search(r"\{.*?\}", text or "", flags=re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
            except json.JSONDecodeError:
                obj = None

    if isinstance(obj, dict):
        complexity = str(obj.get("complexity", "")).strip().lower()
        if complexity in COMPLEXITY_LEVELS:
            try:
                conf = float(obj.get("confidence", 0.5))
            except (TypeError, ValueError):
                conf = 0.5
            conf = min(1.0, max(0.0, conf))
            return RouteDecision(complexity_to_tier(complexity, medium_tier), complexity, conf, True, raw=text)

    return RouteDecision(HIGH, "complex", 0.0, False, raw=text or "")
