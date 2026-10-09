#!/usr/bin/env python3
"""Smoke-check Phase 3F gateway, observability, and authenticated request paths.

Usage:
  python scripts/verify_phase3f.py --base-url http://127.0.0.1:8000
  GATEWAY_TEST_TOKEN='<access-token>' python scripts/verify_phase3f.py

The token is read only from the environment and is never printed. Without it,
public health/status/metrics checks run and authenticated checks are reported skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request


def get(base_url: str, path: str, token: str | None = None, timeout: int = 15):
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(base_url + path, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, response.read().decode("utf-8", errors="replace")


def post_json(base_url: str, path: str, payload: dict, token: str, timeout: int = 420):
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        base_url + path,
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.getenv("GATEWAY_BASE_URL", "http://127.0.0.1:8000"))
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")
    failures = 0
    total_checks = 0

    def check(name, fn):
        nonlocal failures, total_checks
        total_checks += 1
        try:
            result = fn()
            print(f"PASS  {name}: {result}")
        except Exception as exc:  # show test failures but never request/print credentials
            failures += 1
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")

    def health_check():
        status, raw = get(base_url, "/health")
        data = json.loads(raw)
        assert status == 200 and data.get("status") == "ok", data
        return data

    def status_check():
        status, raw = get(base_url, "/api/system/status")
        data = json.loads(raw)
        assert status == 200 and data.get("phase_3f", {}).get("enabled") is True, data
        assert data.get("observability", {}).get("metrics_path") == "/metrics", data
        return "3A–3F feature flags/status present"

    def auth_gate_check():
        request = urllib.request.Request(
            base_url + "/api/chat",
            data=json.dumps({"query": "Phase 3F auth smoke check"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                raise AssertionError(f"unauthenticated request unexpectedly returned HTTP {response.status}")
        except urllib.error.HTTPError as exc:
            assert exc.code in (401, 403), f"expected auth rejection, got HTTP {exc.code}"
            return f"unauthenticated request rejected with HTTP {exc.code}"

    def metrics_check():
        status, raw = get(base_url, "/metrics")
        assert status == 200, f"HTTP {status}"
        required = (
            "routed_to_local_total", "cache_hit_total", "retrieval_latency_ms",
            "rerank_latency_ms", "local_generation_latency_ms", "local_ttft_ms",
            "external_fallback_total", "estimated_cost_avoided_usd_total",
            "usage_ledger_write_failures_total",
        )
        missing = [metric for metric in required if metric not in raw]
        assert not missing, f"missing metrics: {', '.join(missing)}"
        return f"{len(required)} Phase 3F metric families exposed"

    check("gateway health", health_check)
    check("system status", status_check)
    check("unauthenticated chat is rejected", auth_gate_check)
    check("Prometheus metrics", metrics_check)

    token = os.getenv("GATEWAY_TEST_TOKEN", "").strip()
    if not token:
        print("SKIP  authenticated chat/usage checks: set GATEWAY_TEST_TOKEN to a valid access token")
    else:
        def chat_check():
            status, data = post_json(
                base_url, "/api/chat",
                {"query": "What is Chandrayaan-3?", "use_cache": False, "explain": True},
                token,
            )
            assert status == 200 and data.get("answer"), data
            assert data.get("decision", {}).get("tier"), data
            assert data.get("total_latency_ms", 0) >= 0, data
            return f"route={data['decision']['tier']}, total_ms={data.get('total_latency_ms')}"

        def usage_check():
            status, raw = get(base_url, "/api/me/usage", token=token)
            data = json.loads(raw)
            assert status == 200 and "queries_used" in data, data
            return f"queries_used={data['queries_used']}"

        check("authenticated chat", chat_check)
        check("usage ledger endpoint", usage_check)

    print(f"\nPhase 3F smoke-check result: {total_checks - failures}/{total_checks} passed; {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
