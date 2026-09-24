"""Evaluation utilities for computing cost-vs-quality trade-off curves."""
from __future__ import annotations
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

def metrics_at(p_strong: np.ndarray, y_true: np.ndarray, threshold: float, cost_ratio: float = 20.0, scores: Optional[np.ndarray] = None) -> Dict[str, Any]:
    pred_high = (p_strong >= threshold).astype(int)
    n = len(y_true)
    if n == 0: return {}

    tp = int(np.sum((pred_high == 1) & (y_true == 1)))
    fp = int(np.sum((pred_high == 1) & (y_true == 0)))
    fn = int(np.sum((pred_high == 0) & (y_true == 1)))
    tn = int(np.sum((pred_high == 0) & (y_true == 0)))

    accuracy = (tp + tn) / n
    total_cost_routed = np.sum(pred_high * cost_ratio + (1 - pred_high) * 1.0)
    baseline_max_cost = n * cost_ratio
    
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision_high": float(tp / (tp + fp) if (tp + fp) > 0 else 0.0),
        "recall_high": float(tp / (tp + fn) if (tp + fn) > 0 else 0.0),
        "pct_strong": float(np.mean(pred_high)),
        "cost_savings": float(1.0 - (total_cost_routed / baseline_max_cost)),
        "quality": float(np.mean(np.where(pred_high == 1, 5.0, scores)) / 5.0 if scores is not None else accuracy),
    }

def cost_quality_curve(p_strong: np.ndarray, y_true: np.ndarray, cost_ratio: float = 20.0, scores: Optional[np.ndarray] = None) -> pd.DataFrame:
    return pd.DataFrame([metrics_at(p_strong, y_true, thr, cost_ratio, scores) for thr in np.linspace(0.0, 1.0, 101)])

def curve_summary(df: pd.DataFrame) -> Dict[str, float]:
    return {
        "apgr": round(float(np.trapz(df["quality"], df["pct_strong"])), 4),
        "lift_area": round(float(np.trapz(df["accuracy"], df["threshold"])), 4)
    }
