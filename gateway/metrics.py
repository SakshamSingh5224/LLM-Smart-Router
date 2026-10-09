"""Prometheus metrics for MVP 3F gateway routing and local-RAG observability.

Latency histograms use milliseconds to match the JSON API response fields.
Cost metrics are estimates derived from configured per-token pricing, not billing data.
"""
from prometheus_client import Counter, Gauge, Histogram

_POLICY_ACTION_BUCKETS = ("allow", "allow_overage", "downgrade_to_low", "block")
_LATENCY_BUCKETS_MS = (5, 10, 25, 50, 100, 250, 500, 1_000, 2_500, 5_000, 10_000, 30_000, 60_000, 120_000, 300_000, 600_000)

POLICY_DECISION_COUNTER = Counter(
    "policy_decision_total",
    "Routing decisions made by the policy engine, by action",
    ["action"],
)
ROUTED_TO_LOCAL = Counter(
    "routed_to_local_total",
    "Requests served by the local RAG path, including semantic-cache hits",
)
EXTERNAL_FALLBACK = Counter(
    "external_fallback_total",
    "Requests that leave the local RAG path and continue through external routing",
    ["reason"],
)
CACHE_HIT = Counter(
    "cache_hit_total",
    "Responses served from a cache",
    ["cache_type"],
)
RETRIEVAL_LATENCY = Histogram(
    "retrieval_latency_ms",
    "Qdrant retrieval latency in milliseconds",
    buckets=_LATENCY_BUCKETS_MS,
)
RERANK_LATENCY = Histogram(
    "rerank_latency_ms",
    "Reranker wall-clock latency in milliseconds, including first lazy model load",
    buckets=_LATENCY_BUCKETS_MS,
)
RERANK_MODEL_LOAD_LATENCY = Histogram(
    "reranker_model_load_latency_ms",
    "Time spent loading the cross-encoder model, when lazy-loaded",
    buckets=_LATENCY_BUCKETS_MS,
)
LOCAL_GENERATION_LATENCY = Histogram(
    "local_generation_latency_ms",
    "Local Ollama generation latency in milliseconds",
    buckets=_LATENCY_BUCKETS_MS,
)
LOCAL_TTFT = Histogram(
    "local_ttft_ms",
    "Local RAG time-to-first-token in milliseconds",
    buckets=_LATENCY_BUCKETS_MS,
)
ESTIMATED_COST_AVOIDED = Counter(
    "estimated_cost_avoided_usd_total",
    "Estimated external LOW-tier cost avoided by local RAG/cache, using configured pricing",
)
USAGE_LEDGER_WRITE_FAILURES = Counter(
    "usage_ledger_write_failures_total",
    "Usage-ledger writes that failed; chat responses continue but usage may be missing",
)
# This gauge is intentionally not set by request-serving code. Phase 3G evaluation
# should publish a measured value; serving requests alone cannot establish accuracy.
LOCAL_RAG_ACCURACY = Gauge(
    "local_rag_accuracy_ratio",
    "Latest evaluated local-RAG accuracy ratio; populated by the evaluation workflow",
)

# Pre-create labelled series so an empty-but-healthy service still exposes the
# metric families to Prometheus before the first matching request occurs.
for _action in _POLICY_ACTION_BUCKETS:
    POLICY_DECISION_COUNTER.labels(action=_action)
for _cache_type in ("semantic", "exact"):
    CACHE_HIT.labels(cache_type=_cache_type)
for _fallback_reason in ("insufficient_context", "retrieval_error", "generation_error"):
    EXTERNAL_FALLBACK.labels(reason=_fallback_reason)
