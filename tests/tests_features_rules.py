from smartrouter.features import extract_signals
from smartrouter.rules import RulesRouter

def test_extract_signals():
    sig = extract_signals("Write a python function def foo(): return 42")
    assert sig["has_code"] is True
    assert sig["word_count"] > 0

def test_rules_router():
    router = RulesRouter()
    dec = router.route("Write a python function def foo(): return 42")
    assert dec.tier in ("low", "high")
    assert dec.source == "rules_fallback"
