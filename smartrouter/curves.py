"""Evaluation utilities for computing cost-vs-quality trade-off curves."""
from __future__ import annotations

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd


def metrics_at(p_strong: np.ndarray, y_true: np.ndarray, threshold: float, cost_ratio: float = 20.0, scores: Optional[np.ndarray] = None) -> Dict[str, Any]:
    pred_high = (np.array(p_strong) >= threshold).astype(int)
    n = len(y_true)
    if n == 0:
        return {}

    tp = int(np.sum((pred_high == 1) & (y_true == 1)))
    fp = int(np.sum((pred_high == 1) & (y_true == 0)))
    fn = int(np.sum((pred_high == 0) & (y_true == 1)))
    tn = int(np.sum((pred_high == 0) & (y_true == 0)))

    accuracy = (tp + tn) / n
    precision_high = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_high = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    pct_strong = np.mean(pred_high)

    # Cost calculation: cheap = 1 unit, strong = cost_ratio units
    total_cost_routed = np.sum(pred_high * cost_ratio + (1 - pred_high) * 1.0)
    baseline_max_cost = n * cost_ratio
    cost_savings = 1.0 - (total_cost_routed / baseline_max_cost)

    quality = accuracy
    if scores is not None:
        quality = float(np.mean(np.where(pred_high == 1, 5.0, scores))) / 5.0

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision_high": float(precision_high),
        "recall_high": float(recall_high),
        "pct_strong": float(pct_strong),
        "cost_savings": float(cost_savings),
        "quality": float(quality),
    }


def cost_quality_curve(p_strong: np.ndarray, y_true: np.ndarray, cost_ratio: float = 20.0, scores: Optional[np.ndarray] = None) -> pd.DataFrame:
    thresholds = np.linspace(0.0, 1.0, 101)
    rows = [metrics_at(p_strong, y_true, thr, cost_ratio, scores) for thr in thresholds]
    return pd.DataFrame(rows)


def curve_summary(df: pd.DataFrame) -> Dict[str, float]:
    apgr = float(np.trapezoid(df["quality"], df["pct_strong"]))
    lift_area = float(np.trapezoid(df["accuracy"], df["threshold"]))
    return {"apgr": round(apgr, 4), "lift_area": round(lift_area, 4)}
