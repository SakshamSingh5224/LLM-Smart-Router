"""Unit tests for Phase 3F metrics and best-effort usage persistence."""
from gateway.metrics import USAGE_LEDGER_WRITE_FAILURES
from gateway.policy import PolicyEngine


def test_usage_ledger_failure_rolls_back_and_does_not_raise():
    class FailingSession:
        def __init__(self):
            self.added = None
            self.rolled_back = False

        def add(self, row):
            self.added = row

        def commit(self):
            raise RuntimeError("simulated transient database failure")

        def rollback(self):
            self.rolled_back = True

    db = FailingSession()
    before = USAGE_LEDGER_WRITE_FAILURES._value.get()
    result = PolicyEngine(db).record_usage(1, "LOW", 10, 0.0, False)
    assert result is False
    assert db.added is not None
    assert db.rolled_back is True
    assert USAGE_LEDGER_WRITE_FAILURES._value.get() == before + 1
