"""Training, backend selection, and threshold calibration routines."""
from __future__ import annotations
from typing import Dict, List, Tuple, Any
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from .featurizers import build_tfidf_pipeline
from .curves import metrics_at

DEFAULT_PRESETS = {"aggressive": {"target_recall": 0.70}, "balanced": {"target_recall": 0.85}, "conservative": {"target_recall": 0.95}}

def train_router(train_df: pd.DataFrame, val_df: pd.DataFrame, backends: List[str], Cs: List[float], presets: Dict[str, Any] = DEFAULT_PRESETS, cost_ratio: float = 20.0, margin: float = 0.0, max_train: int = 0, seed: int = 42) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if max_train > 0 and len(train_df) > max_train:
        train_df = train_df.sample(n=max_train, random_state=seed)

    X_train, y_train = train_df["prompt"].tolist(), (train_df["tier"] == "high").astype(int).to_numpy()
    X_val, y_val = val_df["prompt"].tolist(), (val_df["tier"] == "high").astype(int).to_numpy()

    best_auc, best_model, best_pipeline, best_C = -1.0, None, None, 1.0

    for C in Cs:
        pipeline = build_tfidf_pipeline()
        clf = LogisticRegression(C=C, max_iter=1000, random_state=seed)
        clf.fit(pipeline.fit_transform(X_train), y_train)
        auc = float(roc_auc_score(y_val, clf.predict_proba(pipeline.transform(X_val))[:, 1]))

        if auc > best_auc:
            best_auc, best_model, best_pipeline, best_C = auc, clf, pipeline, C

    p_val = best_model.predict_proba(best_pipeline.transform(X_val))[:, 1]
    thresholds, val_metrics = {}, {}

    for mode, cfg in presets.items():
        target_rec = min(1.0, cfg["target_recall"] + margin)
        best_thr = min(np.linspace(0.01, 0.99, 100), key=lambda t: abs(metrics_at(p_val, y_val, t, cost_ratio)["recall_high"] - target_rec))
        thresholds[mode] = float(best_thr)
        val_metrics[mode] = metrics_at(p_val, y_val, best_thr, cost_ratio)

    return {
        "pipeline": best_pipeline, "model": best_model, "backend": "tfidf", "C": best_C,
        "val_auc": best_auc, "thresholds": thresholds, "val_metrics": val_metrics,
        "n_train": len(train_df), "n_val": len(val_df), "trained_at": datetime.now(timezone.utc).isoformat(),
    }, {"val_auc": best_auc}
