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
        semantic_threshold=0.82,
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
    assert hit.similarity >= 0.82
    assert hit.decision == decision


def test_semantic_cache_stores_non_local_rag_responses():
    cache = make_cache()
    decision = {
        "tier": "LOW",
        "p_strong": 0.05,
        "threshold": 0.70,
        "confidence": 0.80,
        "mode": "balanced",
        "source": "classifier",
        "reasoning": "simple general question",
        "signals": [],
    }
    cache.put(
        "Explain semantic caching.",
        "balanced",
        "Semantic caching reuses answers for similar questions.",
        "LOW",
        0.05,
        decision,
    )

    hit = cache.semantic_get("What does semantic caching mean?", "balanced")
    assert hit is not None
    assert hit.answer == "Semantic caching reuses answers for similar questions."
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
        semantic_threshold=0.82,
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



def test_language_instruction_does_not_reuse_english_answer():
    cache = make_cache()
    query = "What is Chandrayaan-3?"
    cache.put(query, "balanced", "A lunar mission.", "high", 0.0, {"tier": "high"})
    assert cache.semantic_get(query + " Answer in Hindi.", "balanced") is None


def test_same_language_paraphrase_can_reuse_answer():
    cache = make_cache()
    cache.put("What is Chandrayaan-3? Answer in Hindi.", "balanced", "यह एक चंद्र मिशन है।", "high", 0.0, {"tier": "high"})
    assert cache.semantic_get("Explain Chandrayaan-3. Answer in Hindi.", "balanced") is not None
    assert cache.semantic_get("Explain Chandrayaan-3. Answer in English.", "balanced") is None


def test_format_numbers_and_negation_do_not_cross_match():
    cache = make_cache()
    cache.put("What is Chandrayaan-3?", "balanced", "A mission.", "high", 0.0, {"tier": "high"})
    for query in ["What is Chandrayaan-3? Return JSON.", "What is Chandrayaan-2?", "What is not Chandrayaan-3?"]:
        assert cache.semantic_get(query, "balanced") is None


def test_word_limits_do_not_cross_match():
    cache = make_cache()
    cache.put("Explain Chandrayaan-3 in under 50 words.", "balanced", "A lunar mission.", "high", 0.0, {"tier": "high"})
    assert cache.semantic_get("Explain Chandrayaan-3 in under 10 words.", "balanced") is None


def test_known_instruction_violations_are_not_cached():
    cache = make_cache()
    query = "What is Chandrayaan-3? Answer in Hindi."
    cache.put(query, "balanced", "An English answer.", "high", 0.0, {"tier": "high"})
    assert cache.get(query, "balanced") is None
    assert not cache._redis.data
    query = "Explain Chandrayaan-3 in under 3 words."
    cache.put(query, "balanced", "This is far too long.", "high", 0.0, {"tier": "high"})
    assert cache.get(query, "balanced") is None


def test_legacy_entries_without_constraint_metadata_are_rejected():
    import json
    cache = make_cache()
    cache.put("What is Chandrayaan-3?", "balanced", "A mission.", "high", 0.0, {"tier": "high"})
    key = next(iter(cache._redis.data))
    payload = json.loads(cache._redis.data[key])
    payload.pop("constraints")
    cache._redis.data[key] = json.dumps(payload)
    assert cache.semantic_get("What is Chandrayaan-3?", "balanced") is None


def test_same_word_limit_survives_json_serialization():
    cache = make_cache()
    cache.put("Explain Chandrayaan-3 in under 50 words.", "balanced", "A lunar mission.", "high", 0.0, {"tier": "high"})
    assert cache.semantic_get("Describe Chandrayaan-3 in under 50 words.", "balanced") is not None


def test_normal_instructions_are_not_misread_as_languages():
    from gateway.cache import request_constraints
    assert request_constraints("Write a Python function to remove duplicates.")["languages"] == []
    assert request_constraints("Explain semantic caching in two sentences.")["languages"] == []
    assert request_constraints("Answer in JSON.")["languages"] == []


def test_mostly_english_answer_with_one_hindi_word_is_not_cached():
    cache = make_cache()
    q = "What is Chandrayaan-3? Answer in Hindi."
    cache.put(q, "balanced", "नमस्ते. This is otherwise a long English answer that ignores the language instruction.", "high", 0.0, {"tier": "high"})
    assert cache.get(q, "balanced") is None
