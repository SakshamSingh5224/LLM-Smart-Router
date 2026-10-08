"""MVP 3C: Qdrant Top-10 dense retrieval.

The embedding model is loaded lazily so importing the FastAPI app does not
consume model memory on constrained deployment instances.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from qdrant_client import QdrantClient

if TYPE_CHECKING:
    from fastembed import TextEmbedding

from gateway.settings import load_gateway_settings

log = logging.getLogger("gateway.retriever")


@dataclass(frozen=True)
class RetrievedChunk:
    id: str
    text: str
    dense_score: float
    source: str | None
    filename: str | None
    category: str | None
    page: int | None
    chunk_index: int | None

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "filename": self.filename,
            "category": self.category,
            "page": self.page,
            "chunk_index": self.chunk_index,
        }


@dataclass(frozen=True)
class RetrievalResult:
    query: str
    candidates: list[RetrievedChunk]
    latency_ms: float

    @property
    def best_score(self) -> float:
        return self.candidates[0].dense_score if self.candidates else 0.0

    @property
    def context(self) -> str:
        return "\n\n".join(
            f"[Source: {c.source or 'unknown'}, page: {c.page or 'unknown'}]\n{c.text}"
            for c in self.candidates
        )


class LocalRetriever:
    """Retrieve Top-K chunks from the Phase 3A Qdrant collection."""

    def __init__(self) -> None:
        self.cfg = load_gateway_settings()
        self.client = QdrantClient(
            url=self.cfg.qdrant_url,
            api_key=self.cfg.qdrant_api_key or None,
        )
        self._embedding_model: TextEmbedding | None = None

    @property
    def embedding_model(self) -> TextEmbedding:
        if self._embedding_model is None:
            from fastembed import TextEmbedding
            log.info("Loading retrieval embedding model: %s", self.cfg.embedding_model)
            self._embedding_model = TextEmbedding(model_name=self.cfg.embedding_model)
        return self._embedding_model

    def retrieve(self, query: str, top_k: int | None = None) -> RetrievalResult:
        query = query.strip()
        if not query:
            return RetrievalResult(query="", candidates=[], latency_ms=0.0)

        started = time.perf_counter()
        limit = top_k or self.cfg.retrieval_top_k

        try:
            vector = list(self.embedding_model.embed([query]))[0].tolist()
            response = self.client.query_points(
                collection_name=self.cfg.qdrant_collection,
                query=vector,
                limit=limit,
                with_payload=True,
            )

            candidates: list[RetrievedChunk] = []
            for point in response.points:
                payload = point.payload or {}
                text = str(payload.get("text") or "").strip()
                if not text:
                    continue
                candidates.append(
                    RetrievedChunk(
                        id=str(point.id),
                        text=text,
                        dense_score=float(point.score),
                        source=payload.get("source"),
                        filename=payload.get("filename"),
                        category=payload.get("category"),
                        page=payload.get("page"),
                        chunk_index=payload.get("chunk_index"),
                    )
                )

            return RetrievalResult(
                query=query,
                candidates=candidates,
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        except Exception:
            log.exception("Qdrant retrieval failed")
            return RetrievalResult(
                query=query,
                candidates=[],
                latency_ms=(time.perf_counter() - started) * 1000,
            )

    def search(self, query: str) -> str:
        return self.retrieve(query).context
