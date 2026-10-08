#!/usr/bin/env python3
"""Calibrate MVP 3C reranker scores against labeled local-KB queries.

This does not change the production threshold. It measures score separation
on the actual knowledge base so the 0.78 cutoff can be defended or revised.
"""
import argparse, json, statistics
from gateway.reranker import RetrievalPipeline

DEFAULT_POSITIVE = [
    "What was the objective of Chandrayaan-3?",
    "What is the role of the SHAPE payload on Chandrayaan-3?",
    "What does the LVM3-M4 launch vehicle do in the Chandrayaan-3 mission?",
]
DEFAULT_NEGATIVE = ["What is the recipe for chocolate cake?"]

def run_case(pipeline, label, query):
    result = pipeline.run(query)
    return {
        "label": label, "query": query,
        "best_normalized_score": result.best_score,
        "best_raw_score": result.best_raw_score,
        "sufficient_at_configured_threshold": result.sufficient,
        "threshold": result.threshold,
        "retrieval_latency_ms": result.retrieval_latency_ms,
        "rerank_latency_ms": result.latency_ms,
        "top_sources": [x.chunk.source for x in result.candidates],
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--positive", action="append", default=[])
    p.add_argument("--negative", action="append", default=[])
    args = p.parse_args()
    positives = args.positive or DEFAULT_POSITIVE
    negatives = args.negative or DEFAULT_NEGATIVE

    pipeline = RetrievalPipeline()
    rows = [run_case(pipeline, "positive", q) for q in positives]
    rows += [run_case(pipeline, "negative", q) for q in negatives]
    print(json.dumps(rows, indent=2, default=str))

    pos = [r["best_normalized_score"] for r in rows if r["label"] == "positive"]
    neg = [r["best_normalized_score"] for r in rows if r["label"] == "negative"]
    print("\nSummary")
    if pos:
        print(f"Positive mean={statistics.mean(pos):.4f}, min={min(pos):.4f}")
    if neg:
        print(f"Negative mean={statistics.mean(neg):.4f}, max={max(neg):.4f}")
    if pos and neg:
        print(f"Configured threshold={pipeline.cfg.rerank_relevance_threshold:.2f}")
        print(f"Separation gap (positive min - negative max)={min(pos)-max(neg):.4f}")
        for threshold in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.78, 0.80]:
            tp = sum(s >= threshold for s in pos)
            fn = len(pos) - tp
            fp = sum(s >= threshold for s in neg)
            tn = len(neg) - fp
            print(f"threshold={threshold:.2f}: TP={tp} FN={fn} FP={fp} TN={tn}")
    print("\nNote: sigmoid values normalize CrossEncoder logits to 0..1; they are not calibrated probabilities.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
