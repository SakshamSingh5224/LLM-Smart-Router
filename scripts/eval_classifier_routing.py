#!/usr/bin/env python3
"""Scores saved router artifact against validation or test datasets."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from smartrouter.classifier_router import ClassifierRouter
from smartrouter.config import load_settings


def main() -> None:
    cfg = load_settings()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifact", type=Path, default=cfg.results_dir / "router_classifier.joblib")
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--mode", default=None)
    ap.add_argument("--n", type=int, default=0)
    args = ap.parse_args()

    if not args.artifact.exists():
        sys.exit(f"{args.artifact} not found. Run: python scripts/train_classifier.py")
    router = ClassifierRouter.load(args.artifact)

    data_path = cfg.data_dir / "processed" / f"{args.split}.parquet"
    if not data_path.exists():
        sys.exit(f"{data_path} not found. Run: python scripts/prepare_dataset.py")

    df = pd.read_parquet(data_path)
    if args.n:
        df = df.sample(n=min(args.n, len(df)), random_state=42)

    modes = [args.mode] if args.mode else sorted(router.modes)
    results = {}

    for mode in modes:
        rows = []
        for r in df.itertuples(index=False):
            d = router.route(r.prompt, mode=mode)
            rows.append({"true_tier": r.tier, "pred_tier": d.tier, "latency_ms": d.latency_ms})
        out = pd.DataFrame(rows)
        tp = int(((out.pred_tier == "high") & (out.true_tier == "high")).sum())
        tn = int(((out.pred_tier == "low") & (out.true_tier == "low")).sum())
        n = len(out)

        results[mode] = {
            "n": n,
            "accuracy": round((tp + tn) / n, 4) if n else 0.0,
            "latency_ms_p50": round(float(out["latency_ms"].median()), 3),
        }
        print(f"[{mode:<12}] n={n:<6} accuracy={results[mode]['accuracy']:.3f} latency_p50={results[mode]['latency_ms_p50']}ms")

    out_path = cfg.results_dir / f"eval_classifier_{args.split}.json"
    out_path.write_text(json.dumps({"split": args.split, **results}, indent=2))
    print(f"\nSaved evaluation metrics -> {out_path}")


if __name__ == "__main__":
    main()
