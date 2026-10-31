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


# Phase 3G extends this existing script; it does not create project files.
INSTALL_FILES = (
    "gateway/app.py", "gateway/cache.py", "gateway/auth.py", "gateway/db/database.py",
    "gateway/rag_client.py", "gateway/retriever.py", "scripts/ingest_docs.py",
    "scripts/verify_phase3f.py", "frontend/app.js", "frontend/chat.html",
    "frontend/index.html", "docker/docker-compose.yml", "tests/test_mvp3e_cache.py",
    "tests/test_mvp3d_gateway.py", "tests/test_gateway.py", "tests/test_mvp3d_rag.py",
    "tests/test_integration_policy.py", "tests/test_policy.py", "tests/test_mvp3f_metrics.py",
    "README.md",
)

def install_batch(revision: str, enable_semantic: bool) -> int:
    """Download a pinned review commit, back up tracked edits, validate, deploy."""
    import ast
    import re
    import subprocess
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path
    root = Path.cwd()
    if not re.fullmatch(r"[0-9a-f]{40}", revision or ""):
        raise ValueError("--revision must be a full immutable Git commit SHA")
    if not (root / ".git").exists() or not (root / "gateway/app.py").is_file():
        raise ValueError("Run from the existing llm-smart-router repository root")
    for name in INSTALL_FILES:
        if not (root / name).is_file():
            raise ValueError(f"Expected existing file missing: {name}; nothing changed")
    prefix = f"https://raw.githubusercontent.com/SakshamSingh5224/LLM-Smart-Router/{revision}/"
    def download(name):
        with urllib.request.urlopen(prefix + name, timeout=30) as response:
            content = response.read().decode("utf-8")
        if name.endswith(".py"):
            ast.parse(content, filename=name)
        return name, content
    with ThreadPoolExecutor(max_workers=6) as pool:
        downloaded = list(pool.map(download, INSTALL_FILES))
    backup = subprocess.check_output(["git", "stash", "create", "Before Phase 3G batch"], text=True).strip()
    if backup:
        subprocess.run(["git", "stash", "store", "-m", "Before Phase 3G batch", backup], check=True)
        print("Existing tracked edits backed up in Git stash; working tree was not reset.", flush=True)
    for name, content in downloaded:
        (root / name).write_text(content)
    print("Updated existing files; .env and Vercel tunnel configuration preserved.", flush=True)
    env = dict(os.environ, SEMANTIC_CACHE_ENABLED="false", ENABLE_MVP3_ROUTING="true",
               ENABLE_MVP3_RETRIEVAL="true", DATABASE_URL="sqlite:///:memory:",
               GATEWAY_RATE_LIMIT_PER_MIN="10000", HIGH_API_KEY="offline-test-key",
               GATEWAY_API_KEY="", PYTHONDONTWRITEBYTECODE="1",
               ROUTER_SERVICE_URL="", ROUTER_MODE="balanced", GATEWAY_CACHE_ENABLED="true")
    # Offline clients must not inherit machine-specific SOCKS proxy settings.
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        env.pop(key, None)
    subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"], env=env, check=True)
    if enable_semantic:
        env_path = root / ".env"
        text = env_path.read_text()
        if re.search(r"(?m)^SEMANTIC_CACHE_ENABLED=", text):
            text = re.sub(r"(?m)^SEMANTIC_CACHE_ENABLED=.*$", "SEMANTIC_CACHE_ENABLED=true", text)
        else:
            text = text.rstrip() + "\nSEMANTIC_CACHE_ENABLED=true\n"
        env_path.write_text(text)
    compose = ["docker", "compose", "--env-file", ".env", "-f", "docker/docker-compose.yml"]
    subprocess.run(compose + ["up", "-d", "--no-build", "--no-deps", "gateway"], check=True)
    import time
    for _ in range(90):
        try:
            status, raw = get("http://127.0.0.1:8000", "/health", timeout=2)
            if status == 200 and json.loads(raw).get("status") == "ok":
                print("Gateway ready. No image rebuild was performed.", flush=True)
                return 0
        except Exception:
            pass
        time.sleep(2)
    print("Gateway did not become ready. Inspect compose logs; keep Cloudflare running.")
    return 1


def phase3g_checks(base_url, token, check, latency_budget_ms):
    """Bounded live checks. No third-PDF tests and no service shutdowns."""
    import re
    session = {}
    question = "According to the Chandrayaan-3 PDF, list payload names only, grouped under Lander, Rover and Propulsion Module. Answer in English."
    def call(query, **options):
        status, data = post_json(base_url, "/api/chat", {"query": query, "mode": "balanced", **options}, token)
        assert status == 200, f"HTTP {status}"
        print("      " + json.dumps({k: data.get(k) for k in ("answer", "decision", "cache_hit", "total_latency_ms", "sources")}, ensure_ascii=False), flush=True)
        return data
    def expect_http(payload, expected, bearer=token):
        try:
            status, _ = post_json(base_url, "/api/chat", payload, bearer, timeout=20)
        except urllib.error.HTTPError as exc:
            status = exc.code
        assert status == expected, f"expected HTTP {expected}, got {status}"
        return f"HTTP {status}"
    check("invalid JWT rejected", lambda: expect_http({"query": "test"}, 401, "invalid.jwt.signature"))
    check("empty query rejected", lambda: expect_http({"query": ""}, 422))
    check("invalid forced tier rejected", lambda: expect_http({"query": "test", "force_tier": "invalid"}, 422))
    def local_grounding():
        data = call(question, use_cache=True)
        session["local"] = data
        assert data["decision"]["tier"] == "LOCAL-RAG", data["decision"]
        assert any(s.get("filename") == "chandrayaan3.pdf" and s.get("page") == 7 for s in data.get("sources", [])), "page 7 missing"
        answer = data["answer"].casefold()
        answer = answer.replace("*", "")
        matches = [re.search(pattern, answer) for pattern in (r"\blander(?: payloads?)?\s*:", r"\brover(?: payloads?)?\s*:", r"\bpropulsion(?: module)?(?: payloads?)?\s*:")]
        positions = [match.start() if match else -1 for match in matches]
        assert 0 <= positions[0] < positions[1] < positions[2], "missing/ambiguous payload groups; manual review needed"
        groups = [answer[positions[0]:positions[1]], answer[positions[1]:positions[2]], answer[positions[2]:]]
        assert all(x in groups[0] for x in ("rambha", "chaste", "ilsa")), "lander grouping failed"
        assert all(x in groups[1] for x in ("apxs", "libs")), "rover grouping failed"
        assert "shape" in groups[2] and not any(x in groups[2] for x in ("apxs", "libs", "chaste", "rambha")), "SHAPE grouping failed"
        return "payload groups and PDF citation passed (narrow check, not full faithfulness evaluation)"
    check("local PDF grounding", local_grounding)
    def exact_hit():
        data = call(question, use_cache=True)
        session["exact"] = data
        assert data["cache_hit"] is True, "repeat did not hit cache"
        assert data["router_latency_ms"] == 0 and data["generation_latency_ms"] == 0, "cache did not skip work"
        return f"total_ms={data['total_latency_ms']}"
    check("exact cache skips retrieval/generation", exact_hit)
    def cache_latency():
        data = session.get("exact")
        assert data is not None, "exact-cache check unavailable"
        assert data["total_latency_ms"] <= latency_budget_ms, f"{data['total_latency_ms']} ms exceeds budget {latency_budget_ms} ms"
        return f"within {latency_budget_ms} ms"
    check("cache end-to-end latency target", cache_latency)
    def paraphrase_hit():
        _, raw = get(base_url, "/api/system/status")
        if not json.loads(raw).get("phase_3e", {}).get("enabled"):
            raise AssertionError("semantic cache disabled; cannot mark semantic phase complete")
        data = call("Using the Chandrayaan-3 PDF, name instruments only, grouped under Lander, Rover and Propulsion Module. Answer in English.", use_cache=True)
        assert data["cache_hit"] is True, "paraphrase did not hit semantic cache"
        return "compatible paraphrase reused answer"
    check("semantic paraphrase reuse", paraphrase_hit)
    def hindi_guard():
        data = call(question.replace("Answer in English.", "Answer in Hindi."), use_cache=True)
        assert re.search(r"[\u0900-\u097f]", data["answer"]), "Hindi instruction not followed"
        assert data["decision"]["tier"] == "high", "unsupported local language was not sent to HIGH"
        return "Hindi output without English-cache collision"
    check("language isolation", hindi_guard)
    ncert = "According to NCERT Class 12 Physics, define quantisation of charge, give q = ne, explain n, and explain why quantisation can be ignored macroscopically. Answer in English in under 100 words."
    def ncert_check(force):
        data = call(ncert, use_cache=False, force_tier=force)
        assert data["cache_hit"] is False
        answer = data["answer"].casefold()
        assert "integer" in answer or "integral" in answer, "integer definition missing"
        assert "direction of" not in answer, "charge-sign/direction confusion"
        assert len(data["answer"].split()) < 100, "word limit exceeded"
        assert not re.search(r"1\s*(?:mc|millicoulomb).*?10\s*(?:\^|\*\*)?\s*13", answer), "milli/micro unit error"
        if force:
            assert data["decision"]["tier"] == force and data["decision"]["source"] == "forced_tier"
        else:
            assert data["decision"]["tier"] == "LOCAL-RAG"
        return "narrow math/length checks passed; printed answer still needs human faithfulness review"
    check("NCERT local answer", lambda: ncert_check(None))
    check("forced HIGH provenance and NCERT answer", lambda: ncert_check("high"))
    def insufficient():
        data = call("According to the Chandrayaan-3 PDF, give its final audited expenditure by supplier invoice number. Answer in English.", use_cache=False)
        assert data["decision"]["tier"] in ("low", "high"), "insufficient retrieval did not fall back"
        assert "insufficient" in (data["decision"].get("reasoning") or "").lower(), "missing fallback reason"
        assert not data.get("sources"), "unsupported document-source claim"
        assert not re.search(r"(?:INR|₹|Rs\.?|invoice\s*#)\s*\d", data["answer"], re.I), "potential fabricated expenditure/invoice"
        return "fallback without fabricated invoice figures"
    check("insufficient-context fallback", insufficient)
    def stream_check():
        body = json.dumps({"query": "Reply with OK only.", "force_tier": "high", "use_cache": False}).encode()
        request = urllib.request.Request(base_url + "/api/chat/stream", data=body, headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"}, method="POST")
        with urllib.request.urlopen(request, timeout=120) as response:
            raw = response.read().decode()
        assert "event: meta" in raw and "event: delta" in raw and "event: done" in raw
        assert raw.index("event: meta") < raw.index("event: delta") < raw.rindex("event: done")
        assert '"source": "forced_tier"' in raw and '"tier": "high"' in raw
        return "SSE order and forced-tier metadata valid"
    check("SSE streaming", stream_check)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.getenv("GATEWAY_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--phase3g", action="store_true", help="Run the bulk Phase 3G live suite; excludes third PDF")
    parser.add_argument("--login", action="store_true", help="Prompt for existing credentials; never print/store token")
    parser.add_argument("--install", action="store_true", help="Install a pinned batch into existing files and deploy without build")
    parser.add_argument("--revision", help="Full immutable commit SHA for --install")
    parser.add_argument("--enable-semantic", action="store_true", help="Enable semantic cache after offline checks pass")
    parser.add_argument("--latency-budget-ms", type=float, default=2000)
    args = parser.parse_args()
    if args.install:
        return install_batch(args.revision, args.enable_semantic)
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
    if not token and args.login:
        import getpass
        email = input("Existing account email: ").strip()
        password = getpass.getpass("Password (hidden): ")
        _, auth = post_json(base_url, "/api/auth/login", {"email": email, "password": password}, "", timeout=30)
        password = None
        token = auth.get("access_token", "")
        if not token:
            raise RuntimeError("Login returned no access token")
    if not token:
        print("SKIP  authenticated chat/usage checks: use --login or GATEWAY_TEST_TOKEN")
        if args.phase3g:
            failures += 1
            total_checks += 1
            print("FAIL  Phase 3G cannot be completed without authenticated checks")
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

        if args.phase3g:
            phase3g_checks(base_url, token, check, args.latency_budget_ms)
        else:
            check("authenticated chat", chat_check)
        check("usage ledger endpoint", usage_check)

    print(f"\nValidation result: {total_checks - failures}/{total_checks} passed; {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

