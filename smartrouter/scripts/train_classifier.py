#!/usr/bin/env python3
import argparse, json, joblib, sys, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from smartrouter.config import load_settings
from smartrouter.curves import cost_quality_curve, curve_summary, metrics_at
from smartrouter.training import DEFAULT_PRESETS, train_router
from smartrouter.classifier_router import ClassifierRouter

def main():
    cfg = load_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--Cs", nargs="+", type=float, default=[0.3, 1.0, 3.0])
    ap.add_argument("--out", type=Path, default=cfg.results_dir / "router_classifier.joblib")
    args = ap.parse_args()

    train_df = pd.read_parquet(cfg.data_dir / "processed/train.parquet")
    val_df = pd.read_parquet(cfg.data_dir / "processed/val.parquet")
    test_df = pd.read_parquet(cfg.data_dir / "processed/test.parquet")

    artifact, _ = train_router(train_df, val_df, backends=["tfidf"], Cs=args.Cs)
    
    router = ClassifierRouter(artifact)
    p_test = router.predict_proba(test_df["prompt"].tolist())
    y_test = (test_df["tier"] == "high").astype(int).to_numpy()
    
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.out, compress=3)
    print(f"Saved artifact -> {args.out}")

if __name__ == "__main__": main()
