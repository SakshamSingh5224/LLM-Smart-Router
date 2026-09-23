#!/usr/bin/env python3
"""Phase 1 exit-criteria check.

  1. Local SLM is deployed (Ollama up, router model installed)
  2. SLM router answers under the latency budget (warm p50) with parseable output
  3. LOW tier is reachable
  4. HIGH tier is reachable via API
  5. Routing dataset is prepared

Exit code 0 only if everything passes.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402

from smartrouter.clients import OllamaClient, make_high_client, make_low_client  # noqa: E402
from smartrouter.config import load_settings  # noqa: E402
from smartrouter.router import SLMRouter  # noqa: E402
from smartrouter.utils import percentile  # noqa: E402

PING = [{"role": "user", "content": "Reply with the single word: pong"}]
results: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str) -> bool:
    results.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
    return ok


def http_err(e: Exception) -> str:
    if isinstance(e, httpx.HTTPStatusError):
        hint = {401: "invalid API key", 403: "model not allowed for this key", 404: "wrong model name/base URL",
                429: "free-tier rate limit hit - wait a minute"}.get(e.response.status_code, "")
        return f"HTTP {e.response.status_code} {hint} {e.response.text[:160]}".strip()
    return f"{type(e).__name__}: {e}"


def main() -> int:
    cfg = load_settings()
    print(f"Router SLM : {cfg.router_model} @ {cfg.ollama_host}")
    print(f"Low tier   : {cfg.low_provider}/{cfg.low_model}")
    print(f"High tier  : {cfg.high_model} @ {cfg.high_base_url}\n")

    # 1 + 2 -------------------------------------------------------------
    ollama = OllamaClient(cfg.ollama_host, cfg.router_model)
    up = record("1. Ollama running", ollama.is_up(), cfg.ollama_host)
    installed = False
    if up:
        installed = ollama.has_model()
        record("1. Router model installed", installed,
               cfg.router_model if installed else f"missing -> run: ollama pull {cfg.router_model}")
    if installed:
        try:
            router = SLMRouter(ollama)
            router.route("warm-up")
            samples = ["hi", "What is 17 * 23?", "Refactor this recursive parser to be iterative and explain the trade-offs.",
                       "Give me three name ideas for a coffee shop.", "Explain how B-trees stay balanced."]
            decisions = [router.route(q) for q in samples * 2]
            lat = [d.latency_ms for d in decisions]
            p50 = percentile(lat, 50)
            parsed = sum(d.parse_ok for d in decisions)
            record("2. Router latency (warm p50)", p50 <= cfg.router_latency_budget_ms,
                   f"{p50:.0f} ms (budget {cfg.router_latency_budget_ms:.0f} ms)")
            record("2. Router output parseable", parsed == len(decisions), f"{parsed}/{len(decisions)} valid JSON decisions")
        except Exception as e:  # noqa: BLE001
            record("2. Router inference", False, http_err(e))

    # 3 -----------------------------------------------------------------
    try:
        low = make_low_client(cfg)
        if cfg.low_provider == "ollama":
            if not low.has_model():
                raise RuntimeError(f"model missing -> ollama pull {cfg.low_model}")
            res = low.chat(PING, max_tokens=10)
        else:
            res = low.chat(PING, max_tokens=50)
        record("3. Low tier reachable", bool(res.text), f"'{res.text[:40]}' in {res.latency_ms:.0f} ms")
    except Exception as e:  # noqa: BLE001
        record("3. Low tier reachable", False, http_err(e))

    # 4 -----------------------------------------------------------------
    if not cfg.high_api_key or cfg.high_api_key.startswith("paste_"):
        record("4. High tier reachable", False, "HIGH_API_KEY not set in .env (free key: https://console.groq.com/keys)")
    else:
        try:
            high = make_high_client(cfg)
            available = high.list_models()
            if cfg.high_model not in available:
                record("4. High model listed", False,
                       f"{cfg.high_model} not offered to your key. Available: {', '.join(available[:12])}")
            else:
                res = high.chat(PING, max_tokens=200)  # gpt-oss reasons first; leave room
                note = "" if res.text else " (empty text: reasoning used the budget; endpoint is reachable)"
                record("4. High tier reachable", True, f"'{res.text[:40]}' in {res.latency_ms:.0f} ms{note}")
        except Exception as e:  # noqa: BLE001
            record("4. High tier reachable", False, http_err(e))

    # 5 -----------------------------------------------------------------
    proc = cfg.data_dir / "processed"
    have = [n for n in ("train", "val", "test") if (proc / f"{n}.parquet").exists()]
    record("5. Dataset prepared", len(have) == 3,
           f"found {have}" if len(have) == 3 else "run: python scripts/prepare_dataset.py")

    failed = [r for r in results if not r[1]]
    print("\n" + ("PHASE 1 EXIT CRITERIA MET" if not failed else f"{len(failed)} check(s) failed - fix the FAIL lines above"))
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
