import os
os.environ["DATABASE_URL"] = "sqlite:///:memory:"

import pytest
from unittest.mock import patch
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import sessionmaker

# Setup test database engine before importing app
test_engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

import gateway.db.database as db_mod
import gateway.app as app_mod
from gateway.db.database import Base
from gateway.db.models import Policy, UserPolicy, UsageLedger
from gateway.app import app, get_db

# Force app and database modules to use the test engine/session
db_mod.engine = test_engine
db_mod.SessionLocal = TestingSessionLocal
app_mod.engine = test_engine
app_mod.SessionLocal = TestingSessionLocal

def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

@pytest.fixture(autouse=True)
def setup_database():
    # Create all tables on the test engine prior to each test execution
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)

@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        yield c

@patch("gateway.app.collect_stream")
@patch("gateway.app.get_decision")
def test_policy_enforcement_workflow(mock_get_decision, mock_collect_stream, client):
    # Mock classifier routing decision
    mock_get_decision.return_value = ({
        "tier": "high",
        "p_strong": 0.85,
        "threshold": 0.2,
        "confidence": 0.9,
        "mode": "balanced",
        "source": "classifier",
        "reasoning": "High complexity query."
    }, 15.0)

    # Mock LLM stream response
    class MockStats:
        text = "Paris is the capital of France."
        latency_ms = 120.0
        tokens_est = 25
        error = None
        ttft_ms = 40.0
    mock_collect_stream.return_value = MockStats()

    # 1. Register a test user
    email = "policy_test_user@example.com"
    password = "SecurePassword123!"
    reg_res = client.post("/api/auth/register", json={"email": email, "password": password})
    assert reg_res.status_code == 200

    # 2. Login to receive JWT token
    login_res = client.post("/api/auth/login", json={"email": email, "password": password})
    assert login_res.status_code == 200
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 3. Adjust the user's default policy threshold to 2 queries for testing exhaustion
    db = TestingSessionLocal()
    try:
        free_policy = db.query(Policy).filter(Policy.name == "free").first()
        assert free_policy is not None, "Default policies must be seeded on registration."
        free_policy.query_threshold = 2
        free_policy.action_on_exhaustion = "downgrade_to_low"
        db.commit()
    finally:
        db.close()

    # 4. Query 1: Should route normally (Not Downgraded)
    res1 = client.post("/api/chat", json={"query": "What is the capital of France?", "use_cache": False}, headers=headers)
    assert res1.status_code == 200
    data1 = res1.json()
    assert data1["downgraded"] is False
    assert data1["decision"]["tier"] == "high"

    # 5. Query 2: Should route normally (Not Downgraded, quota limit = 2 reached)
    res2 = client.post("/api/chat", json={"query": "Explain quantum physics.", "use_cache": False}, headers=headers)
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["downgraded"] is False

    # 6. Query 3: Quota exhausted -> Should be force-downgraded to LOW tier by Policy Engine
    res3 = client.post("/api/chat", json={"query": "What is 2+2?", "use_cache": False}, headers=headers)
    assert res3.status_code == 200
    data3 = res3.json()
    assert data3["downgraded"] is True
    assert data3["decision"]["tier"] == "low"
    assert "Monthly quota exhausted" in data3["decision"]["reasoning"]

    # 7. Verify usage ledger records all 3 requests
    db = TestingSessionLocal()
    try:
        ledger_entries = db.query(UsageLedger).all()
        assert len(ledger_entries) == 3
        assert ledger_entries[2].was_downgraded is True
    finally:
        db.close()
