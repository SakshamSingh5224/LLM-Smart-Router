import pytest
from gateway.judge import QueryJudge

@pytest.fixture
def judge():
    return QueryJudge()

def test_local_kb_intent(judge):
    decision = judge.classify_intent("Tell me about the ISRO Chandrayaan mission.")
    assert decision.route == "LOCAL_KB"
    assert decision.intent_score > 0.70
    assert "isro" in decision.matched_signals
    assert "chandrayaan" in decision.matched_signals

def test_general_external_intent(judge):
    decision = judge.classify_intent("What is the capital of France?")
    assert decision.route == "EXTERNAL_LLM"
    assert decision.intent_score < 0.70
    assert len(decision.matched_signals) == 0

def test_ambiguous_intent(judge):
    decision = judge.classify_intent("What is the budget for the space program?")
    # Does not explicitly match the strict "indian budget" signal
    assert decision.route == "EXTERNAL_LLM"
    assert decision.intent_score < 0.70

def test_empty_query(judge):
    decision = judge.classify_intent("   ")
    assert decision.route == "EXTERNAL_LLM"
    assert decision.intent_score == 0.0
    assert len(decision.matched_signals) == 0
