#!/usr/bin/env python3
"""Evaluate the trained router on test.parquet (never used for training/calibration).

    python scripts/eval_router_test.py

Writes:
    results/test_cost_quality_curve.csv
    results/test_metrics.json
    results/test_cost_quality_curve.png   (if matplotlib is available)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402

from smartrouter.classifier_router import ClassifierRouter  # noqa: E402
from smartrouter.config import load_settings  # noqa: E402
from smartrouter.curves import cost_quality_curve, curve_summary, metrics_at  # noqa: E402


def main() -> None:
    cfg = load_settings()
    art_path = cfg.results_dir / "router_artifact.joblib"
    test_path = cfg.data_dir / "processed" / "test.parquet"
    if not art_path.exists():
        sys.exit(f"{art_path} not found. Run: python scripts/train_router.py")
    if not test_path.exists():
        sys.exit(f"{test_path} not found. Run: python scripts/prepare_dataset.py")

    artifact = joblib.load(art_path)
    router = ClassifierRouter(artifact)
    test = pd.read_parquet(test_path)
    y = (test["tier"] == "high").astype(int).to_numpy()
    print(f"Scoring {len(test):,} held-out test prompts with backend={artifact['backend']} ...")
    p = router.predict_proba(test["prompt"].tolist())

    curve = cost_quality_curve(p, y, cost_ratio=artifact["cost_ratio"], scores=test["mixtral_score"].to_numpy())
    summary = curve_summary(curve)

    preset_metrics = {
        mode: metrics_at(p, y, thr, artifact["cost_ratio"], test["mixtral_score"].to_numpy())
        for mode, thr in artifact["thresholds"].items()
    }

    cfg.results_dir.mkdir(exist_ok=True)
    curve.to_csv(cfg.results_dir / "test_cost_quality_curve.csv", index=False)
    result = {
        "n_test": int(len(test)), "base_rate_high": float(y.mean()), "val_auc": artifact["val_auc"],
        "curve_summary": summary, "preset_metrics": preset_metrics,
        "always_low": metrics_at(p, y, 1.01, artifact["cost_ratio"]),
        "always_high": metrics_at(p, y, -0.01, artifact["cost_ratio"]),
    }
    (cfg.results_dir / "test_metrics.json").write_text(json.dumps(result, indent=2, default=float))

    print("\n=== Cost-vs-quality on the held-out TEST split ===")
    print(f"n={len(test):,}  base rate (needs strong)={y.mean():.1%}  APGR={summary['apgr']:.3f} "
          f"(0.5=random, 1.0=perfect)")
    for mode, m in preset_metrics.items():
        print(f"  [{mode:<12}] routed_to_strong={m['pct_strong']:.1%}  quality={m['quality']:.3f}  "
              f"cost_savings_vs_always_strong={m['cost_savings']:.1%}  accuracy={m['accuracy']:.3f}  "
              f"recall_high={m['recall_high']:.3f}")
    al = result["always_low"]
    print(f"  [always_low  ] quality={al['quality']:.3f} (naive baseline; 0% cost)")

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(curve["pct_strong"], curve["quality"], label="router")
        ax.plot(curve["pct_strong"], curve["random_quality"], "--", label="random routing")
        for mode, m in preset_metrics.items():
            ax.scatter([m["pct_strong"]], [m["quality"]], zorder=5)
            ax.annotate(mode, (m["pct_strong"], m["quality"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel("Share of prompts routed to the strong tier")
        ax.set_ylabel("Quality (good-enough rate)")
        ax.set_title("Cost vs. quality - held-out test split")
        ax.legend()
        fig.tight_layout()
        fig.savefig(cfg.results_dir / "test_cost_quality_curve.png", dpi=140)
        print(f"\nSaved plot: {cfg.results_dir / 'test_cost_quality_curve.png'}")
    except ImportError:
        print("\n(matplotlib not installed - skipped the plot; the CSV/JSON have the full curve)")

    print(f"Saved {cfg.results_dir / 'test_cost_quality_curve.csv'} and {cfg.results_dir / 'test_metrics.json'}")


if __name__ == "__main__":
    main()
