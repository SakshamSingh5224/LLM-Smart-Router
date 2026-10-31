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



def test_batch_installer_does_not_write_when_download_is_invalid(tmp_path, monkeypatch):
    import io
    import runpy
    import urllib.request
    from pathlib import Path
    import pytest
    namespace = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/verify_phase3f.py"))
    files = namespace["INSTALL_FILES"]
    (tmp_path / ".git").mkdir()
    for name in files:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("original content\n")
    monkeypatch.chdir(tmp_path)
    def download(request, **kwargs):
        return io.BytesIO(b"invalid python syntax !" if str(request).endswith("gateway/app.py") else b"pass\n")
    monkeypatch.setattr(urllib.request, "urlopen", download)
    with pytest.raises(SyntaxError):
        namespace["install_batch"]("a" * 40, False)
    assert all((tmp_path / name).read_text() == "original content\n" for name in files)


def test_batch_installer_requires_immutable_revision():
    import runpy
    from pathlib import Path
    import pytest
    namespace = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/verify_phase3f.py"))
    with pytest.raises(ValueError, match="immutable"):
        namespace["install_batch"]("main", False)
