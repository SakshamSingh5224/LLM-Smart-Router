"""Train + calibrate the routing classifier.

Flow
 1. features for train / val  (TF-IDF and, if available, BGE embeddings)
 2. logistic regression per backend, C picked by validation ROC-AUC
 3. best backend wins (val AUC)
 4. thresholds for the cost/quality presets are calibrated on VAL
    (cheapest operating point that still meets each quality target)
 5. everything is bundled into one artifact dict (joblib-able)

The TEST split is never touched here.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

from .curves import DEFAULT_COST_RATIO, cost_quality_curve, curve_summary, metrics_at, pick_threshold
from .featurizers import make_featurizer

# quality target = required "good-enough rate"; higher target -> more strong-tier calls
DEFAULT_PRESETS = {"aggressive": 0.90, "balanced": 0.95, "conservative": 0.98}
ARTIFACT_SCHEMA = 1


def _labels(df: pd.DataFrame) -> np.ndarray:
    return (df["tier"] == "high").astype(int).to_numpy()


def _subsample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if not n or n >= len(df):
        return df
    frac = n / len(df)
    return df.groupby("tier", group_keys=False).sample(frac=frac, random_state=seed).reset_index(drop=True)


def train_router(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    backends: Sequence[str] = ("tfidf", "embed"),
    Cs: Sequence[float] = (0.3, 1.0, 3.0),
    featurizer_opts: Optional[dict] = None,
    presets: Optional[dict] = None,
    cost_ratio: float = DEFAULT_COST_RATIO,
    margin: float = 0.0,
    max_train: int = 0,
    seed: int = 42,
    cache_dir: Optional[Path] = None,
    log: Callable[[str], None] = print,
) -> tuple[dict, dict]:
    presets = presets or DEFAULT_PRESETS
    featurizer_opts = featurizer_opts or {}
    train_df = _subsample(train_df, max_train, seed)
    y_tr, y_va = _labels(train_df), _labels(val_df)
    x_tr_text, x_va_text = train_df["prompt"].tolist(), val_df["prompt"].tolist()
    log(f"train={len(train_df):,} (strong needed {y_tr.mean():.1%})  val={len(val_df):,} (strong needed {y_va.mean():.1%})")

    results, best = [], None
    for kind in backends:
        try:
            feat = make_featurizer(kind, **featurizer_opts.get(kind, {}))
            log(f"\n[{kind}] featurizing...")
            feat.fit(x_tr_text)
            cp = (lambda split: str(Path(cache_dir) / f"emb_{split}_{len(train_df)}.npz")) if (cache_dir and kind == "embed") else (lambda split: None)
            x_tr = feat.transform(x_tr_text, cache_path=cp("train"))
            x_va = feat.transform(x_va_text, cache_path=cp("val"))
        except ImportError as e:
            log(f"[{kind}] skipped (missing dependency: {e}). Install it or drop this backend.")
            continue

        solver = "liblinear" if sp.issparse(x_tr) else "lbfgs"
        for C in Cs:
            clf = LogisticRegression(C=C, solver=solver, max_iter=1000)
            clf.fit(x_tr, y_tr)
            p = clf.predict_proba(x_va)[:, 1]
            auc, ll = float(roc_auc_score(y_va, p)), float(log_loss(y_va, p))
            log(f"[{kind}] C={C:<4} val AUC={auc:.4f}  logloss={ll:.4f}")
            results.append({"backend": kind, "C": C, "val_auc": auc, "val_logloss": ll})
            if best is None or auc > best["auc"]:
                best = {"auc": auc, "kind": kind, "C": C, "feat": feat, "clf": clf, "p_val": p}

    if best is None:
        raise RuntimeError("No backend could be trained (all were skipped).")
    log(f"\nBest: backend={best['kind']} C={best['C']} val AUC={best['auc']:.4f}")

    # ---- calibrate thresholds on validation -------------------------------
    p_val = best["p_val"]
    scores_val = val_df["mixtral_score"].to_numpy() if "mixtral_score" in val_df else None
    curve = cost_quality_curve(p_val, y_va, cost_ratio, scores=scores_val)
    thresholds = {m: pick_threshold(curve, q, margin) for m, q in presets.items()}
    val_metrics = {
        m: metrics_at(p_val, y_va, t, cost_ratio, scores_val) | {"target_quality": presets[m]}
        for m, t in thresholds.items()
    }
    for m, r in val_metrics.items():
        log(f"  val [{m:<12}] thr={thresholds[m]:.3f} strong={r['pct_strong']:.1%} quality={r['quality']:.3f} "
            f"acc={r['accuracy']:.3f} recall_high={r['recall_high']:.3f} savings={r['cost_savings']:.1%}")

    artifact = {
        "schema": ARTIFACT_SCHEMA,
        "backend": best["kind"],
        "C": best["C"],
        "featurizer": best["feat"],
        "clf": best["clf"],
        "thresholds": thresholds,
        "presets": presets,
        "default_mode": "balanced" if "balanced" in thresholds else next(iter(thresholds)),
        "cost_ratio": cost_ratio,
        "label_definition": "tier=high when mixtral_score < WEAK_SCORE_THRESHOLD (see data/processed/stats.json)",
        "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "val_auc": best["auc"],
        "val_metrics": val_metrics,
        "sklearn_version": sklearn.__version__,
    }
    report = {
        "candidates": results,
        "best": {"backend": best["kind"], "C": best["C"], "val_auc": best["auc"]},
        "thresholds": thresholds,
        "val_metrics": val_metrics,
        "val_curve_summary": curve_summary(curve),
        "val_base_rate_high": float(y_va.mean()),
    }
    return artifact, {"report": report, "val_curve": curve}
