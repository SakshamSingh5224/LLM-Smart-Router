"""Response caching with exact in-process lookup and MVP 3E Redis semantic cache.

The exact cache preserves the existing gateway behavior.  For LOCAL_KB queries,
MVP 3E adds a Redis-backed semantic cache using the same BAAI/bge-small-en-v1.5
embedding model as retrieval.  Redis failures are intentionally fail-open: a
request simply continues through normal retrieval/generation.
"""
from __future__ import annotations

import json
import logging
import math
import re
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Optional

from prometheus_client import Counter

log = logging.getLogger("gateway.cache")

_WS_RE = re.compile(r"\s+")
SEMANTIC_KEY_PREFIX = "llm-router:semantic-cache:"
SEMANTIC_CACHE_HIT_COUNTER = Counter(
    "semantic_cache_hits_total",
    "Semantic cache hits",
    ["mode"],
)
SEMANTIC_CACHE_MISS_COUNTER = Counter(
    "semantic_cache_misses_total",
    "Semantic cache misses",
    ["mode"],
)
SEMANTIC_CACHE_UNAVAILABLE_COUNTER = Counter(
    "semantic_cache_unavailable_total",
    "Semantic cache unavailable/fail-open events",
)
SEMANTIC_CACHE_WRITE_COUNTER = Counter(
    "semantic_cache_writes_total",
    "Semantic cache writes",
)
SEMANTIC_CACHE_INVALIDATION_COUNTER = Counter(
    "semantic_cache_invalidations_total",
    "Semantic cache invalidations",
)


def normalize(text: str) -> str:
    return _WS_RE.sub(" ", (text or "").strip().lower())


@dataclass
class CacheEntry:
    answer: str
    tier: str
    p_strong: float
    created_at: float
    decision: Optional[dict[str, Any]] = None
    similarity: Optional[float] = None
    sources: list[dict[str, Any]] = field(default_factory=list)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return -1.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return -1.0
    return dot / (norm_a * norm_b)


class ResponseCache:
    """Exact LRU/TTL cache plus a Redis semantic cache.

    ``get``/``put`` retain the original exact-match interface.  ``semantic_get``
    and ``semantic_put`` are used only on the LOCAL_KB path, so enabling MVP 3E
    does not change LOW/HIGH routing semantics.
    """

    def __init__(
        self,
        max_size: int = 500,
        ttl_s: float = 3600.0,
        *,
        redis_url: str = "",
        semantic_enabled: bool = True,
        semantic_threshold: float = 0.92,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
    ):
        self.max_size, self.ttl_s = max_size, ttl_s
        self.semantic_enabled = semantic_enabled
        self.semantic_threshold = semantic_threshold
        self.embedding_model_name = embedding_model

        self._store: "OrderedDict[str, CacheEntry]" = OrderedDict()
        self._lock = Lock()
        self.hits = self.misses = 0

        self._redis = None
        self._embedding_model = None
        self._semantic_lock = Lock()

        if self.semantic_enabled and redis_url:
            try:
                import redis
                self._redis = redis.Redis.from_url(
                    redis_url,
                    decode_responses=True,
                    socket_connect_timeout=0.2,
                    socket_timeout=0.2,
                    health_check_interval=30,
                )
            except Exception:
                log.exception("Failed to initialize Redis semantic cache")
                self._redis = None

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
            self._store.move_to_end(key)
            self.hits += 1
            return entry

    def put(
        self,
        query: str,
        mode: str,
        answer: str,
        tier: str,
        p_strong: float,
        decision: Optional[dict[str, Any]] = None,
        sources: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        key = self._key(query, mode)
        entry = CacheEntry(
            answer, tier, p_strong, time.time(), decision=decision,
            sources=list(sources or []),
        )
        with self._lock:
            self._store[key] = entry
            self._store.move_to_end(key)
            while len(self._store) > self.max_size:
                self._store.popitem(last=False)

        if tier == "LOCAL-RAG":
            self.semantic_put(query, mode, entry)

    def warm_semantic_model(self) -> bool:
        """Load the semantic embedding model before serving requests."""
        if not self.semantic_enabled:
            return False

        try:
            self._get_embedding_model()
            log.info(
                "Semantic-cache embedding model warmed: %s",
                self.embedding_model_name,
            )
            return True
        except Exception:
            log.exception(
                "Semantic-cache embedding warm-up failed; "
                "continuing with lazy initialization"
            )
            return False

    def _get_embedding_model(self):
        if self._embedding_model is None:
            with self._semantic_lock:
                if self._embedding_model is None:
                    from fastembed import TextEmbedding
                    log.info(
                        "Loading semantic-cache embedding model: %s",
                        self.embedding_model_name,
                    )
                    self._embedding_model = TextEmbedding(
                        model_name=self.embedding_model_name
                    )
        return self._embedding_model

    def _embed(self, query: str) -> list[float]:
        vector = list(self._get_embedding_model().embed([query]))[0]
        return vector.tolist() if hasattr(vector, "tolist") else [float(x) for x in vector]

    @staticmethod
    def _serialize(entry: CacheEntry, query: str, vector: list[float], mode: str) -> dict[str, Any]:
        return {
            "query": query,
            "mode": mode,
            "query_embedding": vector,
            "answer": entry.answer,
            "tier": entry.tier,
            "p_strong": entry.p_strong,
            "decision": entry.decision or {},
            "sources": entry.sources,
            "created_at": entry.created_at,
        }

    def semantic_get(self, query: str, mode: str) -> Optional[CacheEntry]:
        """Return the closest valid Redis entry when similarity >= threshold."""
        if not self.semantic_enabled or self._redis is None:
            return None

        try:
            query_vector = self._embed(query)
            best: Optional[CacheEntry] = None
            best_score = -1.0

            pattern = f"{SEMANTIC_KEY_PREFIX}{mode}:*"
            # MVP cache size is intentionally bounded. Redis SCAN avoids blocking
            # Redis while keeping the implementation dependency-light.
            for key in self._redis.scan_iter(match=pattern, count=max(10, self.max_size)):
                raw = self._redis.get(key)
                if not raw:
                    continue
                try:
                    payload = json.loads(raw)
                    created_at = float(payload.get("created_at", 0.0))
                    if self.ttl_s > 0 and time.time() - created_at > self.ttl_s:
                        self._redis.delete(key)
                        continue
                    score = _cosine_similarity(query_vector, payload["query_embedding"])
                    if score > best_score:
                        best_score = score
                        best = CacheEntry(
                            answer=str(payload.get("answer", "")),
                            tier=str(payload.get("tier", "LOCAL-RAG")),
                            p_strong=float(payload.get("p_strong", 0.0)),
                            created_at=created_at,
                            decision=payload.get("decision") or None,
                            similarity=score,
                            sources=payload.get("sources") or [],
                        )
                except (TypeError, ValueError, json.JSONDecodeError, KeyError):
                    log.warning("Ignoring malformed semantic cache entry: %s", key)

            if best is not None and best_score >= self.semantic_threshold:
                SEMANTIC_CACHE_HIT_COUNTER.labels(mode=mode).inc()
                return best

            SEMANTIC_CACHE_MISS_COUNTER.labels(mode=mode).inc()
            return None
        except Exception:
            # Redis, embedding, and serialization errors must never take down the
            # primary RAG path.
            SEMANTIC_CACHE_UNAVAILABLE_COUNTER.inc()
            log.warning("Semantic cache lookup failed; continuing without cache", exc_info=True)
            return None

    def semantic_put(self, query: str, mode: str, entry: CacheEntry) -> bool:
        if not self.semantic_enabled or self._redis is None:
            return False
        try:
            vector = self._embed(query)
            key = f"{SEMANTIC_KEY_PREFIX}{mode}:{uuid.uuid4().hex}"
            payload = self._serialize(entry, query, vector, mode)
            # EXPIRE is the authoritative Redis TTL. The timestamp is retained for
            # deterministic tests and stale-entry cleanup if TTL is disabled.
            self._redis.set(key, json.dumps(payload), ex=max(1, int(self.ttl_s)))
            SEMANTIC_CACHE_WRITE_COUNTER.inc()
            return True
        except Exception:
            SEMANTIC_CACHE_UNAVAILABLE_COUNTER.inc()
            log.warning("Semantic cache write failed; continuing without cache", exc_info=True)
            return False

    def invalidate(self, query: Optional[str] = None, mode: Optional[str] = None) -> int:
        """Invalidate one semantic entry family or the entire semantic cache.

        For a query, all entries under the selected mode are scanned and removed
        because semantic entries are UUID-keyed.
        """
        if self._redis is None:
            return 0
        deleted = 0
        try:
            if query is None:
                pattern = f"{SEMANTIC_KEY_PREFIX}{mode}:*" if mode else f"{SEMANTIC_KEY_PREFIX}*"
                keys = list(self._redis.scan_iter(match=pattern, count=max(10, self.max_size)))
                if keys:
                    deleted = int(self._redis.delete(*keys))
            else:
                # Exact query invalidation removes semantically cached answers whose
                # original query normalized to the supplied text.
                pattern = f"{SEMANTIC_KEY_PREFIX}{mode or '*'}:*"
                target = normalize(query)
                for key in self._redis.scan_iter(match=pattern, count=max(10, self.max_size)):
                    raw = self._redis.get(key)
                    if not raw:
                        continue
                    try:
                        payload = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if normalize(str(payload.get("query", ""))) == target:
                        deleted += int(self._redis.delete(key))
            if deleted:
                SEMANTIC_CACHE_INVALIDATION_COUNTER.inc(deleted)
            return deleted
        except Exception:
            SEMANTIC_CACHE_UNAVAILABLE_COUNTER.inc()
            log.warning("Semantic cache invalidation failed", exc_info=True)
            return 0

    def semantic_stats(self) -> dict[str, Any]:
        return {
            "enabled": self.semantic_enabled,
            "redis_configured": self._redis is not None,
            "threshold": self.semantic_threshold,
        }

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "size": len(self._store),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
            **self.semantic_stats(),
        }
