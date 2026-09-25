import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import fastapi  # noqa: F401
    from fastapi.testclient import TestClient
    HAVE_FASTAPI = True
except ImportError:
    HAVE_FASTAPI = False

# Point the app at a scratch artifact path (missing -> degrades to rules router,
# which is enough to exercise the whole request path without training anything)
os.environ.setdefault("ROUTER_ARTIFACT_PATH", str(Path(tempfile.gettempdir()) / "no_such_router.joblib"))
os.environ.setdefault("HIGH_API_KEY", "test-key-not-used")
os.environ.setdefault("GATEWAY_LOG_PATH", str(Path(tempfile.gettempdir()) / "test_gateway_log.jsonl"))


@unittest.skipUnless(HAVE_FASTAPI, "fastapi not installed")
class GatewayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from gateway import app as gw
        from gateway.tier_clients import StreamChunk

        class FakeClient:
            def __init__(self, reply="ok from fake model"):
                self.reply = reply

            async def stream_chat(self, messages, max_tokens=512, temperature=0.7):
                for word in self.reply.split(" "):
                    yield StreamChunk(delta=word + " ")
                yield StreamChunk(done=True)

        gw.low_client = FakeClient("cheap answer")
        gw.high_client = FakeClient("expensive answer")
        cls.gw = gw
        cls.client = TestClient(gw.app)

    def test_health(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertIn(r.json()["status"], ("ok", "degraded"))

    def test_chat_routes_and_answers(self):
        r = self.client.post("/api/chat", json={"query": "hi there", "use_cache": False})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIn(body["decision"]["tier"], ("low", "high"))
        self.assertTrue(body["answer"])

    def test_force_tier_selects_the_right_backend(self):
        r_low = self.client.post("/api/chat", json={"query": "anything", "force_tier": "low", "use_cache": False})
        r_high = self.client.post("/api/chat", json={"query": "anything", "force_tier": "high", "use_cache": False})
        self.assertIn("cheap answer", r_low.json()["answer"])
        self.assertIn("expensive answer", r_high.json()["answer"])

    def test_empty_query_rejected(self):
        r = self.client.post("/api/chat", json={"query": ""})
        self.assertEqual(r.status_code, 422)

    def test_cache_hit_on_repeat(self):
        payload = {"query": "cache me please", "use_cache": True}
        r1 = self.client.post("/api/chat", json=payload)
        r2 = self.client.post("/api/chat", json=payload)
        self.assertFalse(r1.json()["cache_hit"])
        self.assertTrue(r2.json()["cache_hit"])
        self.assertEqual(r1.json()["answer"], r2.json()["answer"])

    def test_stream_endpoint_emits_meta_delta_done(self):
        with self.client.stream("POST", "/api/chat/stream",
                                json={"query": "stream please", "use_cache": False}) as r:
            body = "".join(r.iter_text())
        self.assertIn("event: meta", body)
        self.assertIn("event: delta", body)
        self.assertIn("event: done", body)
        self.assertLess(body.index("event: meta"), body.index("event: delta"))
        self.assertLess(body.index("event: delta"), body.rindex("event: done"))

    def test_upstream_error_becomes_502(self):
        from gateway.tier_clients import StreamChunk

        class BoomClient:
            async def stream_chat(self, messages, max_tokens=512, temperature=0.7):
                yield StreamChunk(error="upstream is down")

        old = self.gw.low_client
        self.gw.low_client = BoomClient()
        try:
            r = self.client.post("/api/chat", json={"query": "x", "force_tier": "low", "use_cache": False})
            self.assertEqual(r.status_code, 502)
        finally:
            self.gw.low_client = old


@unittest.skipUnless(HAVE_FASTAPI, "fastapi not installed")
class RouterServiceTests(unittest.TestCase):
    def setUp(self):
        from router_service import app as rs

        self.client = TestClient(rs.app)

    def test_health_and_route(self):
        self.assertEqual(self.client.get("/health").status_code, 200)
        r = self.client.post("/route", json={"query": "hello", "explain": True})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIn(body["tier"], ("low", "high"))
        self.assertIsNotNone(body["reasoning"])

    def test_empty_query_rejected(self):
        self.assertEqual(self.client.post("/route", json={"query": "  "}).status_code, 422)


if __name__ == "__main__":
    unittest.main()
