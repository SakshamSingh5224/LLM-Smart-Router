import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("ROUTER_ARTIFACT_PATH", str(Path(tempfile.gettempdir()) / "no_such_router_3d.joblib"))
os.environ.setdefault("GATEWAY_LOG_PATH", str(Path(tempfile.gettempdir()) / "test_gateway_3d_log.jsonl"))
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from gateway.rag_client import LOCAL_RAG
from gateway.tier_clients import StreamChunk


def _result(sufficient=True):
    chunk = SimpleNamespace(
        source="isro.pdf", page=12, text="Chandrayaan-3 launched on July 14, 2023.",
        dense_score=0.9, category="isro", filename="isro.pdf", chunk_index=0,
        metadata={"source": "isro.pdf", "page": 12},
    )
    return SimpleNamespace(
        candidates=[SimpleNamespace(chunk=chunk, rerank_score=0.91, raw_score=2.3)],
        context="[Source: isro.pdf, page: 12]\nChandrayaan-3 launched on July 14, 2023." if sufficient else "",
        sufficient=sufficient, threshold=0.70, best_score=0.91 if sufficient else 0.50,
        latency_ms=5.0,
    )


@pytest.fixture
def gateway_client(monkeypatch):
    from fastapi.testclient import TestClient
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    import gateway.db.database as db_mod
    from gateway import app as gw
    from gateway.db.models import User

    # Give this test its own process-safe in-memory database.
    old_engine = db_mod.engine
    old_session_local = db_mod.SessionLocal

    test_engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    testing_session_local = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=test_engine,
    )

    db_mod.engine = test_engine
    db_mod.SessionLocal = testing_session_local

    # All ORM models are already imported through gateway.app.
    db_mod.Base.metadata.create_all(bind=test_engine)

    db = testing_session_local()
    user = User(email="phase3d-test@example.com", password_hash="unused")
    db.add(user)
    db.commit()
    db.refresh(user)
    user_id = user.id
    db.close()

    old_overrides = dict(gw.app.dependency_overrides)

    def fake_current_user():
        db = testing_session_local()
        try:
            return db.query(User).filter(User.id == user_id).first()
        finally:
            db.close()

    gw.app.dependency_overrides[gw.get_db] = lambda: testing_session_local()
    gw.app.dependency_overrides[gw.get_current_user] = fake_current_user

    class FakeExternal:
        async def stream_chat(self, messages, max_tokens=512, temperature=0.7):
            yield StreamChunk(delta="external fallback")
            yield StreamChunk(done=True)

    class FakeRag:
        async def generate(self, request, max_tokens=512, temperature=0.2):
            from gateway.tier_clients import StreamStats
            return StreamStats("local grounded", 1.0, 0.2, 3, None)

        def stream(self, request, max_tokens=512, temperature=0.2):
            async def gen():
                yield StreamChunk(delta="local grounded")
                yield StreamChunk(done=True)
            return gen()

    old = (
        gw.low_client,
        gw.high_client,
        gw.local_rag_client,
        gw.retrieval_pipeline,
        gw.judge,
    )

    gw.low_client = FakeExternal()
    gw.high_client = FakeExternal()
    gw.local_rag_client = FakeRag()
    gw.judge = SimpleNamespace(
        classify_intent=lambda q: SimpleNamespace(
            route="LOCAL_KB",
            latency_ms=1.0,
        )
    )
    gw.retrieval_pipeline = SimpleNamespace(
        run=lambda q: _result(True)
    )

    client = TestClient(gw.app)

    try:
        yield client, gw, old
    finally:
        (
            gw.low_client,
            gw.high_client,
            gw.local_rag_client,
            gw.retrieval_pipeline,
            gw.judge,
        ) = old

        gw.app.dependency_overrides.clear()
        gw.app.dependency_overrides.update(old_overrides)

        db_mod.engine = old_engine
        db_mod.SessionLocal = old_session_local


def test_local_rag_nonstream(gateway_client):
    client, gw, _ = gateway_client
    r = client.post("/api/chat", json={"query": "When did Chandrayaan-3 launch?", "use_cache": False})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "local grounded"
    assert body["decision"]["tier"] == LOCAL_RAG
    assert body["decision"]["source"] == "local_rag"
    assert body["est_cost_usd"] == 0.0


def test_local_rag_stream(gateway_client):
    client, _, _ = gateway_client
    with client.stream("POST", "/api/chat/stream", json={"query": "When did Chandrayaan-3 launch?"}) as r:
        body = "".join(r.iter_text())
    assert "event: meta" in body
    assert LOCAL_RAG in body
    assert "event: delta" in body
    assert "local grounded" in body
    assert "event: done" in body


def test_local_rag_semantic_cache_hit_skips_retrieval(gateway_client):
    client, gw, _ = gateway_client

    from gateway.cache import CacheEntry

    decision = {
        "tier": LOCAL_RAG,
        "p_strong": 0.0,
        "threshold": 0.70,
        "confidence": 0.91,
        "mode": "balanced",
        "source": "local_rag",
        "reasoning": "cached grounded result",
        "signals": [],
    }

    class FakeSemanticCache:
        def semantic_get(self, query, mode):
            assert query == "Tell me what Chandrayaan-3 is"
            assert mode == "balanced"
            return CacheEntry(
                answer="cached local answer",
                tier=LOCAL_RAG,
                p_strong=0.0,
                created_at=0.0,
                decision=decision,
                similarity=0.95,
            )

        def get(self, query, mode):
            raise AssertionError("exact cache should not be consulted on LOCAL-RAG semantic hit")

    old_cache = gw.cache
    old_retrieval = gw.retrieval_pipeline
    gw.cache = FakeSemanticCache()
    gw.retrieval_pipeline = SimpleNamespace(
        run=lambda q: (_ for _ in ()).throw(AssertionError("Qdrant/retrieval must be skipped on cache hit"))
    )
    try:
        r = client.post(
            "/api/chat",
            json={"query": "Tell me what Chandrayaan-3 is", "use_cache": True},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["answer"] == "cached local answer"
        assert body["cache_hit"] is True
        assert body["decision"]["tier"] == LOCAL_RAG
    finally:
        gw.cache = old_cache
        gw.retrieval_pipeline = old_retrieval


def test_local_rag_stream_semantic_cache_hit_skips_retrieval(gateway_client):
    client, gw, _ = gateway_client

    from gateway.cache import CacheEntry

    decision = {
        "tier": LOCAL_RAG,
        "p_strong": 0.0,
        "threshold": 0.70,
        "confidence": 0.91,
        "mode": "balanced",
        "source": "local_rag",
        "reasoning": "cached grounded result",
        "signals": [],
    }

    class FakeSemanticCache:
        def semantic_get(self, query, mode):
            return CacheEntry(
                answer="cached stream answer",
                tier=LOCAL_RAG,
                p_strong=0.0,
                created_at=0.0,
                decision=decision,
                similarity=0.96,
            )

        def get(self, query, mode):
            raise AssertionError("exact cache should not be consulted on LOCAL-RAG semantic hit")

    old_cache = gw.cache
    old_retrieval = gw.retrieval_pipeline
    gw.cache = FakeSemanticCache()
    gw.retrieval_pipeline = SimpleNamespace(
        run=lambda q: (_ for _ in ()).throw(AssertionError("retrieval must be skipped"))
    )
    try:
        with client.stream(
            "POST",
            "/api/chat/stream",
            json={"query": "What is Chandrayaan-3?", "use_cache": True},
        ) as r:
            body = "".join(r.iter_text())
        assert "event: meta" in body
        assert '"cache_hit": true' in body
        assert "event: delta" in body
        assert "cached stream answer" in body
        assert "event: done" in body
    finally:
        gw.cache = old_cache
        gw.retrieval_pipeline = old_retrieval


def test_phase3f_system_status_endpoint(gateway_client):
    client, _, _ = gateway_client
    response = client.get("/api/system/status")
    assert response.status_code == 200
    body = response.json()
    assert body["gateway"] == "ok"
    assert body["phase_3a"]["qdrant_url_configured"] is True
    assert "enabled" in body["phase_3b"]
    assert "enabled" in body["phase_3c"]
    assert "enabled" in body["phase_3d"]
    assert "enabled" in body["phase_3e"]
    assert body["observability"]["metrics_path"] == "/metrics"
    assert "not dependency connectivity checks" in body["note"]
