"""Lightweight in-process cache for repeated/near-identical queries.

The project plan's stretch goal is a Redis + embedding-similarity semantic
cache. That needs a Redis server and a vector index, which isn't a "free,
zero-infra" fit for local development, so Phase 3 ships an in-memory
normalized-text cache (exact match after whitespace/case normalization) with
TTL + LRU eviction, behind the same interface a Redis-backed cache would use
-- swap this module out later without touching gateway/app.py.
"""
from __future__ import annotations

import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock
from typing import Optional

_WS_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    return _WS_RE.sub(" ", (text or "").strip().lower())


@dataclass
class CacheEntry:
    answer: str
    tier: str
    p_strong: float
    created_at: float


class ResponseCache:
    def __init__(self, max_size: int = 500, ttl_s: float = 3600.0):
        self.max_size, self.ttl_s = max_size, ttl_s
        self._store: "OrderedDict[str, CacheEntry]" = OrderedDict()
        self._lock = Lock()
        self.hits = self.misses = 0

    def _key(self, query: str, mode: str) -> str:
        return f"{mode}::{normalize(query)}"

    def get(self, query: str, mode: str) -> Optional[CacheEntry]:
        key = self._key(query, mode)
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self.misses += 1
                return None
            if time.time() - entry.created_at > self.ttl_s:
                del self._store[key]
                self.misses += 1
                return None
            self._store.move_to_end(key)  # LRU touch
            self.hits += 1
            return entry

    def put(self, query: str, mode: str, answer: str, tier: str, p_strong: float) -> None:
        key = self._key(query, mode)
        with self._lock:
            self._store[key] = CacheEntry(answer, tier, p_strong, time.time())
            self._store.move_to_end(key)
            while len(self._store) > self.max_size:
                self._store.popitem(last=False)

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {"size": len(self._store), "hits": self.hits, "misses": self.misses,
                "hit_rate": round(self.hits / total, 4) if total else 0.0}
