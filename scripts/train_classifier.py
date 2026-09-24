#!/usr/bin/env python3
"""Trains the router classifier on train.parquet, calibrates thresholds on val.parquet, and evaluates test.parquet."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd

from smartrouter.config import load_settings
from smartrouter.curves import cost_quality_curve, curve_summary, metrics_at
from smartrouter.training import DEFAULT_PRESETS, train_router


def load_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"{path} not found. Run: python scripts/prepare_dataset.py")
    return pd.read_parquet(path)


def main() -> None:
    cfg = load_settings()
    proc = cfg.data_dir / "processed"

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backends", nargs="+", default=["tfidf"], help="Feature backends")
    ap.add_argument("--Cs", nargs="+", type=float, default=[0.3, 1.0, 3.0])
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--cost-ratio", type=float, default=20.0)
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=cfg.results_dir / "router_classifier.joblib")
    args = ap.parse_args()

    train_df, val_df, test_df = (load_split(proc / f"{s}.parquet") for s in ("train", "val", "test"))

    print(f"train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")
    artifact, _ = train_router(
        train_df, val_df,
        backends=args.backends, Cs=args.Cs, presets=DEFAULT_PRESETS,
        cost_ratio=args.cost_ratio, margin=args.margin,
        max_train=args.max_train, seed=args.seed
    )

    from smartrouter.classifier_router import ClassifierRouter
    router = ClassifierRouter(artifact)
    p_test = router.predict_proba(test_df["prompt"].tolist())
    y_test = (test_df["tier"] == "high").astype(int).to_numpy()
    scores_test = test_df["mixtral_score"].to_numpy() if "mixtral_score" in test_df else None

    test_curve = cost_quality_curve(p_test, y_test, args.cost_ratio, scores=scores_test)
    test_summary = curve_summary(test_curve)
    test_metrics = {
        mode: metrics_at(p_test, y_test, thr, args.cost_ratio, scores_test) | {"threshold": thr}
        for mode, thr in artifact["thresholds"].items()
    }

    print("\n=== TEST Metrics (Frozen Thresholds from Val) ===")
    for mode, m in test_metrics.items():
        print(f"  [{mode:<12}] thr={m['threshold']:.3f} strong={m['pct_strong']:.1%} "
              f"quality={m['quality']:.3f} acc={m['accuracy']:.3f} "
              f"recall_high={m['recall_high']:.3f} savings={m['cost_savings']:.1%}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.out, compress=3)
    print(f"\nSaved artifact -> {args.out}")

    cfg.results_dir.mkdir(exist_ok=True)
    test_curve.to_csv(cfg.results_dir / "test_curve.csv", index=False)
    report = {
        "backend": artifact["backend"], "C": artifact["C"], "val_auc": artifact["val_auc"],
        "thresholds": artifact["thresholds"], "val_metrics": artifact["val_metrics"],
        "test_metrics": test_metrics, "test_curve_summary": test_summary,
        "n_train": artifact["n_train"], "n_val": artifact["n_val"],
        "n_test": int(len(test_df)), "trained_at": artifact["trained_at"],
    }
    (cfg.results_dir / "train_report.json").write_text(json.dumps(report, indent=2, default=float))
    print("Saved results/train_report.json and results/test_curve.csv")


if __name__ == "__main__":
    main()
