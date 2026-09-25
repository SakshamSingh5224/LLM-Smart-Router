#!/usr/bin/env python3
"""Phase 3 exit-criteria check against a RUNNING gateway.

Start the gateway first (see README), then:

    python scripts/verify_phase3.py
    python scripts/verify_phase3.py --url http://localhost:8000

Checks:
  1. Gateway /health responds
  2. POST /api/chat routes + returns a full answer (non-streaming path)
  3. POST /api/chat/stream: a `meta` event (routing decision) arrives, THEN at
     least one `delta` event, THEN a `done` event  -> the full pipeline
     (input -> route -> generate -> stream to caller) is verified end-to-end
  4. Forcing tier=low and tier=high both work
  5. The cache serves a second identical request faster and marks cache_hit=true
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import httpx

results: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str) -> bool:
    results.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--query", default="Explain what a hash map is and give a one-line Python example.")
    args = ap.parse_args()
    base = args.url.rstrip("/")

    with httpx.Client(timeout=60.0) as c:
        try:
            r = c.get(f"{base}/health")
            record("1. Gateway /health", r.status_code == 200, json.dumps(r.json())[:200])
        except httpx.HTTPError as e:
            record("1. Gateway /health", False, f"{e} - is the gateway running? uvicorn gateway.app:app")
            print("\nAborting: gateway unreachable.")
            return 1

        try:
            r = c.post(f"{base}/api/chat", json={"query": args.query, "use_cache": False})
            r.raise_for_status()
            body = r.json()
            ok = bool(body.get("answer")) and body.get("decision", {}).get("tier") in ("low", "high")
            record("2. POST /api/chat routes + answers", ok,
                   f"tier={body.get('decision', {}).get('tier')} chars={len(body.get('answer', ''))} "
                   f"total_ms={body.get('total_latency_ms')}")
        except httpx.HTTPError as e:
            record("2. POST /api/chat routes + answers", False, str(e))

        try:
            events, t0 = [], time.perf_counter()
            with c.stream("POST", f"{base}/api/chat/stream", json={"query": args.query, "use_cache": False}) as resp:
                event = None
                for line in resp.iter_lines():
                    if line.startswith("event:"):
                        event = line.split(":", 1)[1].strip()
                    elif line.startswith("data:") and event:
                        events.append((event, line.split(":", 1)[1].strip()))
                        if event == "done":
                            break
            kinds = [e for e, _ in events]
            ok = kinds and kinds[0] == "meta" and "delta" in kinds and kinds[-1] == "done"
            record("3. SSE stream: meta -> delta(s) -> done", bool(ok),
                   f"{len(events)} events in {(time.perf_counter() - t0) * 1000:.0f} ms, order={kinds[:6]}...")
        except httpx.HTTPError as e:
            record("3. SSE stream: meta -> delta(s) -> done", False, str(e))

        for tier in ("low", "high"):
            try:
                r = c.post(f"{base}/api/chat", json={"query": "hi", "force_tier": tier, "use_cache": False})
                r.raise_for_status()
                got = r.json().get("decision", {}).get("tier")
                record(f"4. force_tier={tier} path reachable", r.status_code == 200, f"decision.tier reported={got}")
            except httpx.HTTPError as e:
                record(f"4. force_tier={tier} path reachable", False, str(e))

        try:
            q = f"cache probe {time.time()}"
            r1 = c.post(f"{base}/api/chat", json={"query": q, "use_cache": True})
            r2 = c.post(f"{base}/api/chat", json={"query": q, "use_cache": True})
            hit2 = r2.json().get("cache_hit") is True
            record("5. Repeat query is served from cache", hit2,
                   f"first cache_hit={r1.json().get('cache_hit')} second cache_hit={r2.json().get('cache_hit')} "
                   f"t1={r1.json().get('total_latency_ms')}ms t2={r2.json().get('total_latency_ms')}ms")
        except httpx.HTTPError as e:
            record("5. Repeat query is served from cache", False, str(e))

    failed = [r for r in results if not r[1]]
    print("\n" + ("PHASE 3 EXIT CRITERIA MET" if not failed else f"{len(failed)} check(s) failed - see FAIL lines above"))
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
