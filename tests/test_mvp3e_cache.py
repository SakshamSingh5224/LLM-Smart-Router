import math
import threading
import time

from gateway.cache import CacheEntry, ResponseCache


class FakeRedis:
    def __init__(self):
        self.data = {}
        self.expires = {}
        self.lock = threading.Lock()

    def set(self, key, value, ex=None):
        with self.lock:
            self.data[key] = value
            self.expires[key] = time.time() + ex if ex else None
        return True

    def get(self, key):
        with self.lock:
            expiry = self.expires.get(key)
            if expiry is not None and time.time() >= expiry:
                self.data.pop(key, None)
                self.expires.pop(key, None)
                return None
            return self.data.get(key)

    def scan_iter(self, match=None, count=10):
        import fnmatch
        with self.lock:
            keys = list(self.data)
        for key in keys:
            if match is None or fnmatch.fnmatch(key, match):
                if self.get(key) is not None:
                    yield key

    def delete(self, *keys):
        deleted = 0
        with self.lock:
            for key in keys:
                if key in self.data:
                    del self.data[key]
                    self.expires.pop(key, None)
                    deleted += 1
        return deleted


class FakeEmbeddingModel:
    def embed(self, texts):
        for text in texts:
            t = text.lower()
            # Deterministic toy vectors for semantic-cache behavior tests.
            if "chandrayaan" in t:
                yield [1.0, 0.0, 0.0]
            elif "isro" in t:
                yield [0.0, 1.0, 0.0]
            else:
                yield [0.0, 0.0, 1.0]


def make_cache(ttl=3600):
    cache = ResponseCache(
        max_size=10,
        ttl_s=ttl,
        semantic_enabled=True,
        semantic_threshold=0.92,
        embedding_model="BAAI/bge-small-en-v1.5",
    )
    cache._redis = FakeRedis()
    cache._embedding_model = FakeEmbeddingModel()
    return cache


def test_semantic_cache_hit_above_threshold():
    cache = make_cache()
    decision = {
        "tier": "LOCAL-RAG",
        "p_strong": 0.0,
        "threshold": 0.70,
        "confidence": 0.95,
        "mode": "balanced",
        "source": "local_rag",
        "reasoning": "grounded",
        "signals": [],
    }
    cache.put("What is Chandrayaan-3?", "balanced", "It is a lunar mission.", "LOCAL-RAG", 0.0, decision)

    hit = cache.semantic_get("Tell me what Chandrayaan-3 is", "balanced")

    assert hit is not None
    assert hit.answer == "It is a lunar mission."
    assert hit.similarity is not None
    assert hit.similarity >= 0.92
    assert hit.decision == decision


def test_semantic_cache_miss_below_threshold():
    cache = make_cache()
    cache.put(
        "What is Chandrayaan-3?",
        "balanced",
        "lunar mission",
        "LOCAL-RAG",
        0.0,
        {"tier": "LOCAL-RAG"},
    )

    assert cache.semantic_get("What is ISRO?", "balanced") is None


def test_semantic_cache_ttl_expiry():
    cache = make_cache(ttl=1)
    cache.put(
        "What is Chandrayaan-3?",
        "balanced",
        "lunar mission",
        "LOCAL-RAG",
        0.0,
        {"tier": "LOCAL-RAG"},
    )
    key = next(iter(cache._redis.data))
    cache._redis.expires[key] = time.time() - 1

    assert cache.semantic_get("What is Chandrayaan-3?", "balanced") is None
    assert cache._redis.get(key) is None


def test_semantic_cache_invalidation():
    cache = make_cache()
    cache.put("What is Chandrayaan-3?", "balanced", "answer", "LOCAL-RAG", 0.0, {"tier": "LOCAL-RAG"})

    assert cache.invalidate("What is Chandrayaan-3?", "balanced") == 1
    assert cache.semantic_get("What is Chandrayaan-3?", "balanced") is None


def test_redis_unavailable_fails_open():
    cache = ResponseCache(
        semantic_enabled=True,
        semantic_threshold=0.92,
        redis_url="redis://127.0.0.1:1",
    )
    # Force the same failure mode even when redis-py is installed.
    class BrokenRedis:
        def scan_iter(self, **kwargs):
            raise ConnectionError("redis unavailable")

    cache._redis = BrokenRedis()
    cache._embedding_model = FakeEmbeddingModel()

    assert cache.semantic_get("What is Chandrayaan-3?", "balanced") is None


def test_semantic_cache_concurrent_reads():
    cache = make_cache()
    cache.put("What is Chandrayaan-3?", "balanced", "answer", "LOCAL-RAG", 0.0, {"tier": "LOCAL-RAG"})

    results = []
    lock = threading.Lock()

    def worker():
        hit = cache.semantic_get("Tell me what Chandrayaan-3 is", "balanced")
        with lock:
            results.append(hit.answer if hit else None)

    threads = [threading.Thread(target=worker) for _ in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results == ["answer"] * 16


def test_semantic_embedding_model_warmup():
    cache = make_cache()

    assert cache.warm_semantic_model() is True
    assert cache._embedding_model is not None

    # Calling warm-up again must reuse the already-loaded model.
    model = cache._embedding_model
    assert cache.warm_semantic_model() is True
    assert cache._embedding_model is model


def test_semantic_cache_preserves_source_metadata():
    cache = make_cache()
    sources = [{"filename": "isro.pdf", "source": "isro.pdf", "page": 12, "rerank_score": 0.93}]
    cache.put(
        "What is Chandrayaan-3?", "balanced", "It is a lunar mission.",
        "LOCAL-RAG", 0.0, {"tier": "LOCAL-RAG"}, sources=sources,
    )
    hit = cache.semantic_get("What is Chandrayaan-3?", "balanced")
    assert hit is not None
    assert hit.sources == sources
