"""Hand-crafted domain signals extracted directly from query text."""
from __future__ import annotations

import re
from typing import Dict, Any

CODE_PATTERNS = [
    r"```", r"\bdef\s+", r"\bclass\s+", r"\bimport\s+", r"\breturn\b",
    r"\bfunction\b", r"\bconst\b", r"\blet\b", r"\bvar\b", r"\bfor\s*\(",
    r"\bwhile\s*\(", r"\bif\s*\(", r"\{.*\}", r"\bSQL\b", r"\bSELECT\b"
]

MATH_PATTERNS = [
    r"\$", r"\\frac", r"\\sum", r"\\int", r"\bLaTeX\b", r"\bproof\b",
    r"\btheorem\b", r"\bcalculus\b", r"\bmatrix\b", r"[\+\-\*/\^=]{3,}"
]

REASONING_PATTERNS = [
    r"\bprove\b", r"\bexplain why\b", r"\bcompare\b", r"\banalyze\b",
    r"\boptimize\b", r"\bdebug\b", r"\brefactor\b", r"\bstep-by-step\b"
]

def extract_signals(query: str) -> Dict[str, Any]:
    q_lower = query.lower()
    words = query.split()
    word_count = len(words)
    char_count = len(query)

    has_code = any(re.search(pat, query, re.IGNORECASE) for pat in CODE_PATTERNS)
    has_math = any(re.search(pat, query, re.IGNORECASE) for pat in MATH_PATTERNS)
    has_reasoning = any(re.search(pat, q_lower) for pat in REASONING_PATTERNS)

    return {
        "word_count": word_count,
        "char_count": char_count,
        "has_code": bool(has_code),
        "has_math": bool(has_math),
        "has_reasoning": bool(has_reasoning),
        "is_long": word_count > 120,
    }
