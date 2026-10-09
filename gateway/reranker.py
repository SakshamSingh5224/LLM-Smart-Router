"""MVP 3C: Qdrant Top-10 -> bge-reranker-base Top-3."""
from __future__ import annotations
import logging, math, time
from threading import Lock
from dataclasses import dataclass
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder
from gateway.retriever import LocalRetriever, RetrievedChunk, RetrievalResult
from gateway.settings import load_gateway_settings
from gateway.metrics import RERANK_MODEL_LOAD_LATENCY

log = logging.getLogger("gateway.reranker")

@dataclass(frozen=True)
class RerankedChunk:
    chunk: RetrievedChunk
    rerank_score: float
    raw_score: float

@dataclass(frozen=True)
class RerankResult:
    query: str
    candidates: list[RerankedChunk]
    latency_ms: float
    sufficient: bool
    threshold: float
    retrieval_latency_ms: float = 0.0

    @property
    def best_score(self) -> float:
        return self.candidates[0].rerank_score if self.candidates else 0.0

    @property
    def best_raw_score(self) -> float:
        return self.candidates[0].raw_score if self.candidates else 0.0

    @property
    def context(self) -> str:
        return "\n\n".join(
            f"[Source: {x.chunk.source or 'unknown'}, page: {x.chunk.page or 'unknown'}]\n{x.chunk.text}"
            for x in self.candidates
        )

    @property
    def sources(self) -> list[dict]:
        return [{
            **x.chunk.metadata,
            "dense_score": x.chunk.dense_score,
            "rerank_score": x.rerank_score,
            "rerank_raw_score": x.raw_score,
        } for x in self.candidates]

class LocalReranker:
    def __init__(self) -> None:
        self.cfg = load_gateway_settings()
        self._model: CrossEncoder | None = None
        self._model_lock = Lock()
        self._predict_lock = Lock()

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    from sentence_transformers import CrossEncoder
                    load_started = time.perf_counter()
                    log.info("Loading reranker model: %s", self.cfg.reranker_model)
                    self._model = CrossEncoder(
                        self.cfg.reranker_model,
                        max_length=self.cfg.reranker_max_length,
                    )
                    load_ms = (time.perf_counter() - load_started) * 1000
                    RERANK_MODEL_LOAD_LATENCY.observe(max(0.0, load_ms))
                    log.info("Reranker model load: %.1f ms", load_ms)
        return self._model

    @staticmethod
    def _sigmoid(value: float) -> float:
        value = max(-60.0, min(60.0, value))
        return 1.0 / (1.0 + math.exp(-value))

    def rerank(self, query: str, retrieval: RetrievalResult) -> RerankResult:
        started = time.perf_counter()
        candidates = retrieval.candidates[: self.cfg.reranker_candidate_k]
        if not candidates:
            return RerankResult(query=query, candidates=[], latency_ms=0.0,
                                sufficient=False, threshold=self.cfg.rerank_relevance_threshold,
                                retrieval_latency_ms=retrieval.latency_ms)

        pairs = [(query, c.text) for c in candidates]
        # Access the lazy-loaded model before timing predict() so model download /
        # initialization is observable separately from steady-state inference.
        model = self.model
        inference_started = time.perf_counter()
        # A single local cross-encoder inference at a time keeps memory and CPU
        # pressure predictable on the target laptop while other HTTP tasks stay responsive.
        with self._predict_lock:
            raw_scores = model.predict(
                pairs, show_progress_bar=False, batch_size=min(32, len(pairs))
            )
        inference_ms = (time.perf_counter() - inference_started) * 1000

        ranked = sorted(
            (RerankedChunk(c, self._sigmoid(float(score)), float(score))
             for c, score in zip(candidates, raw_scores)),
            key=lambda x: x.rerank_score, reverse=True,
        )
        selected = ranked[: self.cfg.rerank_top_k]
        best = selected[0].rerank_score if selected else 0.0
        log.info("Reranker inference: %.1f ms for %d candidates", inference_ms, len(candidates))
        return RerankResult(
            query=query, candidates=selected,
            latency_ms=(time.perf_counter() - started) * 1000,
            sufficient=best >= self.cfg.rerank_relevance_threshold,
            threshold=self.cfg.rerank_relevance_threshold,
            retrieval_latency_ms=retrieval.latency_ms,
        )

class RetrievalPipeline:
    def __init__(self) -> None:
        self.cfg = load_gateway_settings()
        self.retriever = LocalRetriever()
        self.reranker = LocalReranker()

    def run(self, query: str) -> RerankResult:
        retrieval = self.retriever.retrieve(query, top_k=self.cfg.retrieval_top_k)
        return self.reranker.rerank(query, retrieval)
