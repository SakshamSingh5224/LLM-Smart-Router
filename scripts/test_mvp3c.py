#!/usr/bin/env python3
"""Standalone Phase 3C validation: Qdrant Top-10 -> reranker Top-3."""
import argparse, json
from gateway.reranker import RetrievalPipeline

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    args = parser.parse_args()
    pipeline = RetrievalPipeline()
    result = pipeline.run(args.query)
    print("=" * 72)
    print("MVP 3C - RETRIEVAL + RE-RANKING")
    print("=" * 72)
    print(f"Query: {args.query}")
    print(f"Dense candidates requested: {pipeline.cfg.retrieval_top_k}")
    print(f"Selected context: {len(result.candidates)}")
    print(f"Retrieval latency: {result.retrieval_latency_ms:.1f} ms")
    print(f"Rerank latency: {result.latency_ms:.1f} ms")
    print(f"Best raw rerank score: {result.best_raw_score:.4f}")
    print(f"Best normalized rerank score: {result.best_score:.4f}")
    print(f"Threshold: {result.threshold:.2f}")
    print(f"Sufficient: {result.sufficient}")
    for rank, item in enumerate(result.candidates, 1):
        print("-" * 72)
        print(f"Rank: {rank}")
        print(f"Rerank score: {item.rerank_score:.4f}")
        print(f"Raw score: {item.raw_score:.4f}")
        print(f"Dense score: {item.chunk.dense_score:.4f}")
        print(f"Source: {item.chunk.source}")
        print(f"Page: {item.chunk.page}")
        print(f"Category: {item.chunk.category}")
        print(item.chunk.text[:1500])
    print("\nSources JSON:")
    print(json.dumps(result.sources, indent=2, default=str))
    if not result.sufficient:
        print("\nRESULT: INSUFFICIENT_RETRIEVAL")
        return 2
    print("\nRESULT: PASS")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
