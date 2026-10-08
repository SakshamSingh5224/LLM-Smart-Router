import os, sys
from pathlib import Path
import pytest

os.environ["QDRANT_URL"] = "http://test-qdrant"
os.environ["QDRANT_COLLECTION"] = "test_collection"
os.environ["RETRIEVAL_TOP_K"] = "10"
os.environ["RERANK_TOP_K"] = "3"
os.environ["RERANKER_CANDIDATE_K"] = "10"
os.environ["RERANK_RELEVANCE_THRESHOLD"] = "0.78"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gateway.retriever import LocalRetriever, RetrievalResult, RetrievedChunk
from gateway.reranker import LocalReranker

def make_chunk(i: int, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        id=f"id-{i}", text=text, dense_score=0.9 - i * 0.01,
        source=f"source-{i}.pdf", filename=f"source-{i}.pdf",
        category="isro", page=i + 1, chunk_index=i,
    )

def test_retriever_config_is_top_10():
    retriever = LocalRetriever()
    assert retriever.cfg.retrieval_top_k == 10
    assert retriever.cfg.qdrant_collection == "test_collection"

def test_empty_query_does_not_call_qdrant():
    result = LocalRetriever().retrieve("   ")
    assert result.candidates == []
    assert result.latency_ms == 0.0

def test_reranker_selects_top_3_and_applies_threshold(monkeypatch):
    reranker = LocalReranker()
    class FakeModel:
        def predict(self, pairs, show_progress_bar=False, batch_size=None):
            return [4.0, 3.0, 2.0, 0.0, -1.0]
    monkeypatch.setattr(reranker, "_model", FakeModel())
    retrieval = RetrievalResult(
        query="When did Chandrayaan-3 launch?",
        candidates=[make_chunk(0, "Chandrayaan-3 launched on July 14, 2023."),
                     make_chunk(1, "The mission demonstrated a soft landing."),
                     make_chunk(2, "ISRO operated the mission."),
                     make_chunk(3, "Unrelated context."),
                     make_chunk(4, "Another unrelated context.")],
        latency_ms=2.0,
    )
    result = reranker.rerank(retrieval.query, retrieval)
    assert len(result.candidates) == 3
    assert result.sufficient is True
    assert result.best_score >= 0.78
    assert result.sources[0]["source"] == "source-0.pdf"
    assert result.sources[0]["rerank_raw_score"] == 4.0

def test_reranker_marks_insufficient_context(monkeypatch):
    reranker = LocalReranker()
    class FakeModel:
        def predict(self, pairs, show_progress_bar=False, batch_size=None):
            return [-2.0] * len(pairs)
    monkeypatch.setattr(reranker, "_model", FakeModel())
    retrieval = RetrievalResult(
        query="Unknown",
        candidates=[make_chunk(i, f"chunk {i}") for i in range(10)],
        latency_ms=1.0,
    )
    result = reranker.rerank("Unknown", retrieval)
    assert len(result.candidates) == 3
    assert result.sufficient is False
    assert result.best_score < 0.78
