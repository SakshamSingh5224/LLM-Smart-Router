"""Small stdlib-only helpers."""
from __future__ import annotations

import math
from typing import Sequence


def percentile(values: Sequence[float], p: float) -> float:
    """Linear-interpolated percentile (p in 0..100)."""
    if not values:
        return float("nan")
    xs = sorted(values)
    k = (len(xs) - 1) * p / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return xs[int(k)]
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summarize_ms(values: Sequence[float]) -> dict:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "min": round(min(values), 1),
        "mean": round(sum(values) / len(values), 1),
        "p50": round(percentile(values, 50), 1),
        "p95": round(percentile(values, 95), 1),
        "p99": round(percentile(values, 99), 1),
        "max": round(max(values), 1),
    }
