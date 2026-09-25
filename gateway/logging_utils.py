"""Append-only JSONL logging of routing decisions, latencies and estimated cost.

One line per request. Simple, human-inspectable, greppable, and trivial to
load with `pandas.read_json(path, lines=True)` for the Phase 4 dashboards.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class RequestLog:
    request_id: str
    ts: float
    query_chars: int
    mode: str
    tier: str
    source: str                 # "classifier" | "rules_fallback"
    p_strong: float
    confidence: float
    cache_hit: bool
    router_latency_ms: float
    generation_latency_ms: float
    ttft_ms: Optional[float]
    total_latency_ms: float
    tokens_est: int
    est_cost_usd: float
    error: Optional[str] = None
    signals: list = field(default_factory=list)


class JsonlLogger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, entry: RequestLog) -> None:
        line = json.dumps(asdict(entry), ensure_ascii=False)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def tail(self, n: int = 20) -> list[dict]:
        if not self.path.exists():
            return []
        with open(self.path, encoding="utf-8") as f:
            lines = f.readlines()[-n:]
        return [json.loads(x) for x in lines if x.strip()]


def now_id() -> str:
    return f"{int(time.time() * 1000):x}-{threading.get_ident() % 0xFFFF:04x}"
