import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from gateway.db.database import Base
from gateway.db.models import User, Policy, UserPolicy, UsageLedger
from gateway.policy import PolicyEngine
from smartrouter.labels import LOW, HIGH

@pytest.fixture
def test_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()
    yield db
    db.close()

def test_policy_engine_evaluates_thresholds(test_db):
    # Setup policies
    free_policy = Policy(name="free", query_threshold=2, action_on_exhaustion="downgrade_to_low")
    block_policy = Policy(name="strict", query_threshold=1, action_on_exhaustion="block")
    test_db.add_all([free_policy, block_policy])
    
    # Setup test users
    user_free = User(email="free@test.com", password_hash="hash")
    user_strict = User(email="strict@test.com", password_hash="hash")
    test_db.add_all([user_free, user_strict])
    test_db.commit()

    test_db.add(UserPolicy(user_id=user_free.id, policy_id=free_policy.id))
    test_db.add(UserPolicy(user_id=user_strict.id, policy_id=block_policy.id))
    test_db.commit()

    engine = PolicyEngine(test_db)

    # 1. Test Free User - Under Threshold -> Allow
    decision = engine.evaluate(user_free.id)
    assert decision.action == "allow"
    
    # 2. Add Usage
    engine.record_usage(user_free.id, HIGH, 100, 0.05, False)
    engine.record_usage(user_free.id, HIGH, 100, 0.05, False)
    
    # 3. Test Free User - Over Threshold -> Downgrade
    decision = engine.evaluate(user_free.id)
    assert decision.action == "downgrade_to_low"
    assert decision.is_downgraded is True

    # 4. Test Strict User - Add Usage
    engine.record_usage(user_strict.id, LOW, 50, 0.01, False)
    
    # 5. Test Strict User - Over Threshold -> Block
    decision = engine.evaluate(user_strict.id)
    assert decision.action == "block"



def test_rate_limiter_rejects_burst_without_waiting():
    from gateway.security import TokenBucket
    bucket = TokenBucket(rate_per_min=1, burst=2)
    assert bucket.allow("same-client")
    assert bucket.allow("same-client")
    assert not bucket.allow("same-client")
    assert bucket.allow("different-client")


def test_policy_and_usage_evaluated_in_one_select(test_db):
    from sqlalchemy import event
    policy = Policy(name="one-query", query_threshold=5, tier_scope="high")
    user = User(email="one-query@test.com", password_hash="hash")
    test_db.add_all([policy, user]); test_db.commit()
    test_db.add(UserPolicy(user_id=user.id, policy_id=policy.id)); test_db.commit()
    uid = user.id
    selects = []
    def record(conn, cursor, statement, params, context, executemany):
        if statement.lstrip().upper().startswith("SELECT"):
            selects.append(statement)
    engine = test_db.get_bind()
    event.listen(engine, "before_cursor_execute", record)
    try:
        assert PolicyEngine(test_db).evaluate(uid).action == "allow"
        assert len(selects) == 1
    finally:
        event.remove(engine, "before_cursor_execute", record)


def test_high_only_policy_counts_only_high_usage(test_db):
    policy = Policy(name="high-only", query_threshold=1, tier_scope="high")
    user = User(email="high-only@test.com", password_hash="hash")
    test_db.add_all([policy, user]); test_db.commit()
    test_db.add(UserPolicy(user_id=user.id, policy_id=policy.id)); test_db.commit()
    engine = PolicyEngine(test_db)
    engine.record_usage(user.id, LOW, 10, 0, False)
    assert engine.evaluate(user.id).action == "allow"
    engine.record_usage(user.id, HIGH, 10, 0, False)
    assert engine.evaluate(user.id).is_downgraded


def test_zero_quota_blocks_without_first_allowing_a_request(test_db):
    policy = Policy(name="zero-quota", query_threshold=0, action_on_exhaustion="block")
    user = User(email="zero-quota@test.com", password_hash="hash")
    test_db.add_all([policy, user]); test_db.commit()
    test_db.add(UserPolicy(user_id=user.id, policy_id=policy.id)); test_db.commit()
    assert PolicyEngine(test_db).evaluate(user.id).action == "block"
