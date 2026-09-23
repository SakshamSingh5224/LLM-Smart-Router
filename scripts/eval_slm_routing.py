#!/usr/bin/env python3
"""Zero-shot routing accuracy of the local SLM on held-out RouteLLM prompts.

This validates the core hypothesis of the project ("can a small local model tell
which prompts need the strong tier?") BEFORE we build the rest of the system.

    python scripts/eval_slm_routing.py --n 200

Positive class = "high" (needs the strong model). The number that matters most is
recall_high: the share of hard prompts we correctly escalate (quality risk).
`pct_routed_low` is the cost-saving proxy.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from smartrouter.clients import OllamaClient  # noqa: E402
from smartrouter.config import load_settings  # noqa: E402
from smartrouter.labels import HIGH, LOW  # noqa: E402
from smartrouter.router import SLMRouter  # noqa: E402
from smartrouter.slm_router import complexity_to_tier  # noqa: E402
from smartrouter.utils import summarize_ms  # noqa: E402


def metrics(y_true: list[str], y_pred: list[str]) -> dict:
    n = len(y_true)
    tp = sum(t == HIGH and p == HIGH for t, p in zip(y_true, y_pred))
    fp = sum(t == LOW and p == HIGH for t, p in zip(y_true, y_pred))
    fn = sum(t == HIGH and p == LOW for t, p in zip(y_true, y_pred))
    tn = sum(t == LOW and p == LOW for t, p in zip(y_true, y_pred))
    div = lambda a, b: round(a / b, 4) if b else 0.0  # noqa: E731
    return {
        "accuracy": div(tp + tn, n),
        "precision_high": div(tp, tp + fp),
        "recall_high": div(tp, tp + fn),
        "pct_routed_low": div(tn + fn, n),
        "confusion": {"tp_high": tp, "fp_high": fp, "fn_high": fn, "tn_low": tn},
    }


def main() -> None:
    cfg = load_settings()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=200, help="number of test prompts to score")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--data", type=Path, default=cfg.data_dir / "processed" / "test.parquet")
    args = ap.parse_args()

    if not args.data.exists():
        sys.exit(f"{args.data} not found. Run: python scripts/prepare_dataset.py")
    client = OllamaClient(cfg.ollama_host, cfg.router_model)
    if not client.is_up() or not client.has_model():
        sys.exit(f"Ollama not ready or model missing. Run: ollama pull {cfg.router_model}")

    full = pd.read_parquet(args.data)
    df = full.sample(n=min(args.n, len(full)), random_state=args.seed)
    router = SLMRouter(client)
    router.route("warm-up")

    rows = []
    for i, r in enumerate(df.itertuples(index=False), 1):
        d = router.route(r.prompt)
        rows.append({
            "id": r.id, "true_tier": r.tier, "true_complexity": r.complexity,
            "pred_complexity": d.complexity, "confidence": d.confidence,
            "parse_ok": d.parse_ok, "latency_ms": d.latency_ms,
        })
        if i % 25 == 0 or i == len(df):
            print(f"  scored {i}/{len(df)}")
    out = pd.DataFrame(rows)

    y_true = out["true_tier"].tolist()
    result = {
        "model": cfg.router_model,
        "n": len(out),
        "parse_failure_rate": round(1 - out["parse_ok"].mean(), 4),
        "latency_ms": summarize_ms(out["latency_ms"].tolist()),
        "baselines": {
            "always_low": metrics(y_true, [LOW] * len(out)),
            "always_high": metrics(y_true, [HIGH] * len(out)),
        },
    }
    for medium_tier in (LOW, HIGH):
        # parse failures were already mapped to "complex" -> HIGH (safe default)
        y_pred = [complexity_to_tier(c, medium_tier) for c in out["pred_complexity"]]
        result[f"slm_medium_to_{medium_tier}"] = metrics(y_true, y_pred)

    cfg.results_dir.mkdir(exist_ok=True)
    out.to_csv(cfg.results_dir / "eval_slm_predictions.csv", index=False)
    (cfg.results_dir / "eval_slm_metrics.json").write_text(json.dumps(result, indent=2))

    print("\n=== Zero-shot SLM routing vs. GPT-4-judged labels ===")
    print(f"prompts={result['n']}  parse_failure_rate={result['parse_failure_rate']}  "
          f"latency p50={result['latency_ms']['p50']} ms")
    for name in ("always_low", "always_high"):
        b = result["baselines"][name]
        print(f"{name:<22} acc={b['accuracy']:.3f}  recall_high={b['recall_high']:.3f}  routed_low={b['pct_routed_low']:.3f}")
    for key in ("slm_medium_to_low", "slm_medium_to_high"):
        m = result[key]
        print(f"{key:<22} acc={m['accuracy']:.3f}  recall_high={m['recall_high']:.3f}  "
              f"precision_high={m['precision_high']:.3f}  routed_low={m['pct_routed_low']:.3f}")
    print("\nSaved results/eval_slm_metrics.json and results/eval_slm_predictions.csv")
    print("Read this as a BASELINE. A small zero-shot SLM is often only slightly better than the "
          "majority baseline; Phase 2 trains a classifier on the train split to beat it.")


if __name__ == "__main__":
    main()
