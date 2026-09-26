"""Production hardening for the gateway (Phase 4).

A deployed gateway is reachable by anyone on the internet, and it's holding
your Groq key server-side. Two protections, both dependency-free:

* API key gate  — require `X-API-Key` on /api/* if GATEWAY_API_KEY is set.
  Leave GATEWAY_API_KEY unset for local development (gate is a no-op then).
* Rate limiter  — a simple in-memory token bucket per API key / client IP.
  Not distributed (fine for a single Render instance; if you scale to
  multiple instances, move this to Redis).
"""
from __future__ import annotations

import time
from collections import defaultdict
from threading import Lock

from fastapi import HTTPException, Request


class TokenBucket:
    def __init__(self, rate_per_min: int, burst: int | None = None):
        self.rate_per_sec = rate_per_min / 60.0
        self.capacity = burst or max(rate_per_min, 1)
        self._buckets: dict[str, tuple[float, float]] = {}  # key -> (tokens, last_ts)
        self._lock = Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            tokens, last = self._buckets.get(key, (self.capacity, now))
            tokens = min(self.capacity, tokens + (now - last) * self.rate_per_sec)
            if tokens < 1.0:
                self._buckets[key] = (tokens, now)
                return False
            self._buckets[key] = (tokens - 1.0, now)
            return True


def client_key(req: Request, api_key: str | None) -> str:
    if api_key:
        return f"key:{api_key}"
    fwd = req.headers.get("x-forwarded-for")
    ip = (fwd.split(",")[0].strip() if fwd else None) or (req.client.host if req.client else "unknown")
    return f"ip:{ip}"


def make_guard(required_key: str | None, rate_per_min: int):
    """Returns a FastAPI dependency enforcing both the API key and rate limit."""
    bucket = TokenBucket(rate_per_min)

    async def guard(request: Request):
        supplied = request.headers.get("x-api-key")
        if required_key and supplied != required_key:
            raise HTTPException(status_code=401, detail="missing or invalid X-API-Key")
        key = client_key(request, supplied)
        if not bucket.allow(key):
            raise HTTPException(status_code=429, detail="rate limit exceeded, slow down")

    return guard
