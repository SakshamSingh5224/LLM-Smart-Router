"""Locust load test for the gateway.

Install:  pip install locust   (already in requirements.txt)
Run (UI): locust -f loadtest/locustfile.py --host http://localhost:8000
          then open http://localhost:8089 and set users / spawn rate
Run (headless, ramps 10 -> 100 -> 300 over 3 minutes):
    locust -f loadtest/locustfile.py --host http://localhost:8000 \
      --headless -u 300 -r 5 -t 3m --csv results/loadtest

Set LOCUST_API_KEY to send X-API-Key if the gateway has GATEWAY_API_KEY set.

Each simulated user registers its own account on startup (unique email per
user) rather than sharing one login. This matters for two reasons now that
/api/chat requires auth and is policy-gated:
  1. One shared account would blow through the "free" policy's 50 queries/month
     threshold almost immediately at load-test volume, causing the policy engine
     to downgrade every request to the low tier mid-test - which skews the
     tier-distribution numbers this test is trying to measure, not what we want.
  2. Many distinct users is a more realistic traffic shape anyway.
"""
from __future__ import annotations

import os
import random
import uuid

from locust import HttpUser, between, task

PROMPTS = [
    "hi",
    "What is the capital of Japan?",
    "Summarize the benefits of exercise in two sentences.",
    "Write a Python function to reverse a linked list and explain the time complexity.",
    "Prove that there are infinitely many prime numbers.",
]
API_KEY = os.getenv("LOCUST_API_KEY", "")
PASSWORD = "LoadTest123!"


class GatewayUser(HttpUser):
    wait_time = between(1, 3)

    def on_start(self):
        email = f"loadtest-{uuid.uuid4().hex[:12]}@test.local"
        self.client.post("/api/auth/register", json={"email": email, "password": PASSWORD})
        r = self.client.post("/api/auth/login", json={"email": email, "password": PASSWORD})
        token = r.json().get("access_token", "") if r.status_code == 200 else ""
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}
        if API_KEY:
            self.headers["X-API-Key"] = API_KEY

    @task(5)
    def chat(self):
        query = random.choice(PROMPTS)
        with self.client.post(
            "/api/chat", json={"query": query, "use_cache": False},
            headers=self.headers, name="/api/chat", catch_response=True, timeout=30,
        ) as r:
            if r.status_code != 200:
                r.failure(f"HTTP {r.status_code}: {r.text[:200]}")
            elif not r.json().get("answer"):
                r.failure("empty answer")

    @task(1)
    def health(self):
        self.client.get("/health", name="/health")
