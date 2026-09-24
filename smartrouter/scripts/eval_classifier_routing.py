#!/usr/bin/env python3
import argparse, sys, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from smartrouter.classifier_router import ClassifierRouter
from smartrouter.config import load_settings

def main():
    cfg = load_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=Path, default=cfg.results_dir / "router_classifier.joblib")
    args = ap.parse_args()

    router = ClassifierRouter.load(args.artifact)
    df = pd.read_parquet(cfg.data_dir / "processed/val.parquet")

    for mode in router.modes:
        tp, tn, n = 0, 0, len(df)
        for r in df.itertuples(index=False):
            d = router.route(r.prompt, mode=mode)
            if d.tier == r.tier:
                if d.tier == "high": tp += 1
                else: tn += 1
        print(f"[{mode:<12}] Accuracy: {(tp + tn) / n:.3f}")

if __name__ == "__main__": main()
