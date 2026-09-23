#!/usr/bin/env python3
"""Benchmark the local router SLM: cold start vs warm latency.

    python scripts/benchmark_slm.py            # 30 warm runs
    python scripts/benchmark_slm.py --runs 60

Writes results/latency_slm.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smartrouter.clients import OllamaClient  # noqa: E402
from smartrouter.config import load_settings  # noqa: E402
from smartrouter.router import SLMRouter  # noqa: E402
from smartrouter.utils import summarize_ms  # noqa: E402

PROMPTS = [
    "hi",
    "What is the capital of Australia?",
    "Translate 'good morning' into Spanish.",
    "Explain the difference between TCP and UDP in a few sentences.",
    "Write a Python function that merges overlapping intervals and add unit tests for edge cases.",
    "Prove that the square root of 2 is irrational, then generalise the argument to any non-square integer.",
    "You are a senior SRE. Given this stack trace and the following config snippet, diagnose the "
    "intermittent 502s behind our NGINX ingress and propose a rollout plan with rollback criteria. " * 4,
]


def main() -> None:
    cfg = load_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=30)
    args = ap.parse_args()

    client = OllamaClient(cfg.ollama_host, cfg.router_model)
    if not client.is_up():
        sys.exit(f"Ollama is not reachable at {cfg.ollama_host}. Start it: `sudo systemctl start ollama`")
    if not client.has_model():
        sys.exit(f"Model {cfg.router_model} not installed. Run: ollama pull {cfg.router_model}")

    router = SLMRouter(client)

    print(f"Model: {cfg.router_model}   Budget (warm p50): {cfg.router_latency_budget_ms:.0f} ms")
    print("\n[1/2] Cold start (model evicted from memory first)...")
    client.unload()
    cold = router.route(PROMPTS[1])
    print(f"      cold first-call latency: {cold.latency_ms:.0f} ms")

    print(f"\n[2/2] Warm runs x{args.runs}...")
    router.route(PROMPTS[0])  # extra warm-up, not measured
    lat, parse_fail = [], 0
    for i in range(args.runs):
        d = router.route(PROMPTS[i % len(PROMPTS)])
        lat.append(d.latency_ms)
        parse_fail += 0 if d.parse_ok else 1
    warm = summarize_ms(lat)
    print(f"      warm latency (ms): {warm}")
    print(f"      parse failures   : {parse_fail}/{args.runs}")

    ok = warm["p50"] <= cfg.router_latency_budget_ms
    print(f"\nRESULT: warm p50 {warm['p50']} ms vs budget {cfg.router_latency_budget_ms:.0f} ms -> "
          f"{'PASS' if ok else 'ABOVE BUDGET'}")
    if not ok:
        print("  Tip: use ROUTER_MODEL=qwen2.5:0.5b, or raise ROUTER_LATENCY_BUDGET_MS on CPU. "
              "A BERT-style classifier (Phase 2) brings this to single-digit ms.")

    cfg.results_dir.mkdir(exist_ok=True)
    out = {
        "model": cfg.router_model,
        "cold_ms": round(cold.latency_ms, 1),
        "warm": warm,
        "parse_failures": parse_fail,
        "budget_ms": cfg.router_latency_budget_ms,
        "pass": ok,
    }
    (cfg.results_dir / "latency_slm.json").write_text(json.dumps(out, indent=2))
    print(f"Saved {cfg.results_dir / 'latency_slm.json'}")


if __name__ == "__main__":
    main()
