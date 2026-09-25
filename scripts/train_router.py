#!/usr/bin/env python3
"""Train the routing classifier on train.parquet, calibrate on val.parquet.

    python scripts/train_router.py                       # tfidf, default presets
    python scripts/train_router.py --backends tfidf embed # also try embeddings
    python scripts/train_router.py --max-train 20000      # faster iteration

Writes:
    results/router_artifact.joblib   (loaded by router_service and the gateway)
    results/train_report.json        (candidate scores, calibrated thresholds)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402

from smartrouter.config import load_settings  # noqa: E402
from smartrouter.training import DEFAULT_PRESETS, train_router  # noqa: E402


def main() -> None:
    cfg = load_settings()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backends", nargs="+", default=["tfidf"], choices=["tfidf", "embed"])
    ap.add_argument("--Cs", nargs="+", type=float, default=[0.3, 1.0, 3.0])
    ap.add_argument("--max-train", type=int, default=0, help="subsample train split (0 = use all)")
    ap.add_argument("--cost-ratio", type=float, default=20.0, help="strong-tier cost / cheap-tier cost")
    ap.add_argument("--margin", type=float, default=0.0, help="extra safety margin above each quality target")
    ap.add_argument("--data-dir", type=Path, default=cfg.data_dir / "processed")
    ap.add_argument("--out", type=Path, default=cfg.results_dir / "router_artifact.joblib")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train_p, val_p = args.data_dir / "train.parquet", args.data_dir / "val.parquet"
    if not train_p.exists() or not val_p.exists():
        sys.exit(f"Missing {train_p} or {val_p}. Run: python scripts/prepare_dataset.py")

    print(f"Loading {train_p.name} / {val_p.name} ...")
    train_df, val_df = pd.read_parquet(train_p), pd.read_parquet(val_p)

    cache_dir = cfg.data_dir / "embed_cache"
    artifact, extra = train_router(
        train_df, val_df, backends=args.backends, Cs=args.Cs, presets=DEFAULT_PRESETS,
        cost_ratio=args.cost_ratio, margin=args.margin, max_train=args.max_train,
        seed=args.seed, cache_dir=cache_dir,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.out, compress=3)
    size_kb = args.out.stat().st_size / 1024
    print(f"\nSaved {args.out} ({size_kb:.0f} KB)")

    report_path = args.out.parent / "train_report.json"
    report_path.write_text(json.dumps(extra["report"], indent=2, default=float))
    print(f"Saved {report_path}")

    print("\n=== Beat-the-baseline check (val split) ===")
    print("Compare these numbers with `make eval` (zero-shot SLM) from Phase 1's results/eval_slm_metrics.json.")
    for mode, m in extra["report"]["val_metrics"].items():
        print(f"  [{mode:<12}] accuracy={m['accuracy']:.3f}  recall_high={m['recall_high']:.3f}  "
              f"routed_low={1 - m['pct_strong']:.3f}  quality={m['quality']:.3f}")
    print(f"  val AUC = {artifact['val_auc']:.4f}")


if __name__ == "__main__":
    main()
