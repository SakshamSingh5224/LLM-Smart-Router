"""Locust load test for the gateway.

Install:  pip install locust   (already in requirements.txt)
Run (UI): locust -f loadtest/locustfile.py --host http://localhost:8000
          then open http://localhost:8089 and set users / spawn rate
Run (headless, ramps 10 -> 100 -> 300 over 3 minutes):
    locust -f loadtest/locustfile.py --host http://localhost:8000 \
      --headless -u 300 -r 5 -t 3m --csv results/loadtest

Set LOCUST_API_KEY to send X-API-Key if the gateway has GATEWAY_API_KEY set.
"""
from __future__ import annotations

import os
import random

from locust import HttpUser, between, task

PROMPTS = [
    "hi",
    "What is the capital of Japan?",
    "Summarize the benefits of exercise in two sentences.",
    "Write a Python function to reverse a linked list and explain the time complexity.",
    "Prove that there are infinitely many prime numbers.",
]
API_KEY = os.getenv("LOCUST_API_KEY", "")


class GatewayUser(HttpUser):
    wait_time = between(1, 3)

    def on_start(self):
        self.headers = {"X-API-Key": API_KEY} if API_KEY else {}

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
