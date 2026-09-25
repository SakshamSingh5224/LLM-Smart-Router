"""Cheap hand-crafted prompt features (pure Python + numpy, no ML deps).

Used in two places:
  * as extra numeric inputs to the classifier
  * by the rule-based fallback router (rules.py) and for human-readable "signals"
"""
from __future__ import annotations

import re
from typing import Sequence

import numpy as np

MAX_FEATURE_CHARS = 8000  # bound regex work on huge inputs

CODE_RE = re.compile(
    r"```|\bdef \w+\(|\bclass \w+|\bimport [\w.]+|#include|\bfunction\b|=>|"
    r"\bSELECT\b[^\n]{0,80}\bFROM\b|</?[a-z][a-z0-9]*>|\bpublic\s+static\b|console\.log|"
    r"\breturn\b[^\n]*;|\bfor\s*\(|\bwhile\s*\(",
    re.I,
)
MATH_RE = re.compile(
    r"\d\s*[-+*/^=]\s*\d|\\frac|\\sum|\\int|"
    r"\b(integral|derivative|equation|prove|proof|theorem|solve|sqrt|matrix|probability|"
    r"calculate|formula|algebra|logarithm)\b",
    re.I,
)
FORMAT_RE = re.compile(
    r"\b(json|yaml|xml|csv|markdown|table|bullet points?|step[- ]by[- ]step|word count|"
    r"at least \d+|at most \d+|exactly \d+|no more than \d+|in \d+ (words|sentences|paragraphs))\b",
    re.I,
)
REASON_RE = re.compile(
    r"\b(explain why|compare|contrast|analy[sz]e|design|optimi[sz]e|trade-?offs?|evaluate|"
    r"critique|pros and cons|justify|derive|implement|refactor|debug|architecture|strategy)\b",
    re.I,
)
GREETING_RE = re.compile(r"^\s*(hi|hello|hey|yo|thanks|thank you|ok|okay|yes|no|good (morning|evening|day))\b", re.I)
URL_RE = re.compile(r"https?://\S+")
DIGIT_RE = re.compile(r"\d")
UPPER_RE = re.compile(r"[A-Z]")
SENT_RE = re.compile(r"[.!?]+(\s|$)")

FEATURE_NAMES = [
    "log_chars", "log_words", "log_lines", "has_code", "has_math", "has_format", "has_reasoning",
    "is_greeting", "digit_ratio", "upper_ratio", "nonascii_ratio", "question_marks", "has_url",
    "avg_word_len", "log_sentences", "ends_question", "has_triple_quote",
]


def handcrafted_features(text: str) -> np.ndarray:
    t = (text or "")[:MAX_FEATURE_CHARS]
    n = len(t)
    words = t.split()
    nw = len(words)
    nn = max(n, 1)
    f = [
        np.log1p(n),
        np.log1p(nw),
        np.log1p(t.count("\n") + 1),
        float(bool(CODE_RE.search(t))),
        float(bool(MATH_RE.search(t))),
        float(bool(FORMAT_RE.search(t))),
        float(bool(REASON_RE.search(t))),
        float(bool(GREETING_RE.search(t))),
        len(DIGIT_RE.findall(t)) / nn,
        len(UPPER_RE.findall(t)) / nn,
        (n - len(t.encode("ascii", "ignore"))) / nn,
        min(t.count("?"), 5) / 5.0,
        float(bool(URL_RE.search(t))),
        (sum(len(w) for w in words) / nw) if nw else 0.0,
        np.log1p(len(SENT_RE.findall(t))),
        float(t.rstrip().endswith("?")),
        float('"""' in t or "'''" in t),
    ]
    return np.asarray(f, dtype=np.float32)


def handcrafted_matrix(texts: Sequence[str]) -> np.ndarray:
    if len(texts) == 0:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
    return np.vstack([handcrafted_features(t) for t in texts])


def describe_signals(text: str) -> list[str]:
    """Human-readable reasons, used for the optional `reasoning` field of /route."""
    t = (text or "")[:MAX_FEATURE_CHARS]
    nw = len(t.split())
    out = []
    if CODE_RE.search(t):
        out.append("contains code")
    if MATH_RE.search(t):
        out.append("math/formal reasoning")
    if REASON_RE.search(t):
        out.append("analysis/design verbs")
    if FORMAT_RE.search(t):
        out.append("output-format constraints")
    if nw > 150:
        out.append("long prompt")
    elif nw < 6:
        out.append("very short prompt")
    if GREETING_RE.search(t):
        out.append("greeting/small talk")
    return out
