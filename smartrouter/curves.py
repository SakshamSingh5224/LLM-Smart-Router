"""Cost-vs-quality analysis for a router (numpy/pandas only).

Conventions
-----------
y        : 1 if the prompt NEEDS the strong model ("high"), else 0 ("low")
p        : router score = P(strong needed);  route HIGH when p > threshold
quality  : "good-enough rate" = share of prompts whose final answer is acceptable.
           Routed-high prompts are acceptable by definition (strong model = reference);
           routed-low prompts are acceptable only if the cheap model was good enough.
             quality = 1 - (hard prompts sent to the cheap model) / N
pct_strong : share of prompts sent to the strong tier (the cost driver)
pgr      : performance-gap-recovered, RouteLLM-style:
             (quality - quality_always_low) / (1 - quality_always_low)
rel_cost : cost relative to "always strong", assuming a strong call costs
           `cost_ratio` times a cheap call.
random_quality : quality of a router that sends the same share randomly.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

DEFAULT_COST_RATIO = 20.0


def decision_metrics(
    routed_high,
    y,
    cost_ratio: float = DEFAULT_COST_RATIO,
    scores=None,
) -> dict:
    routed_high = np.asarray(routed_high, dtype=bool)
    y = np.asarray(y).astype(bool)
    n = len(y)
    tp = int((routed_high & y).sum())
    fp = int((routed_high & ~y).sum())
    fn = int((~routed_high & y).sum())
    tn = int((~routed_high & ~y).sum())
    pct = float(routed_high.mean()) if n else 0.0
    q_low = float((~y).mean()) if n else 0.0
    quality = 1.0 - fn / n if n else 0.0
    div = lambda a, b: a / b if b else 0.0  # noqa: E731
    recall, spec, prec = div(tp, tp + fn), div(tn, tn + fp), div(tp, tp + fp)
    row = {
        "pct_strong": pct,
        "quality": quality,
        "pgr": div(quality - q_low, 1.0 - q_low) if q_low < 1.0 else 1.0,
        "rel_cost": (pct * cost_ratio + (1.0 - pct)) / cost_ratio,
        "random_quality": q_low + pct * (1.0 - q_low),
        "accuracy": div(tp + tn, n),
        "balanced_accuracy": (recall + spec) / 2.0,
        "recall_high": recall,
        "specificity": spec,
        "precision_high": prec,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }
    row["cost_savings"] = 1.0 - row["rel_cost"]
    row["lift_over_random"] = row["quality"] - row["random_quality"]
    if scores is not None:
        s = np.minimum(np.asarray(scores, dtype=float), 5.0) / 5.0
        row["score_retention"] = float(np.where(routed_high, 1.0, s).mean())
    return row


def metrics_at(p, y, threshold: float, cost_ratio: float = DEFAULT_COST_RATIO, scores=None) -> dict:
    row = decision_metrics(np.asarray(p) > threshold, y, cost_ratio, scores)
    row["threshold"] = float(threshold)
    return row


def cost_quality_curve(p, y, cost_ratio: float = DEFAULT_COST_RATIO, n_points: int = 201, scores=None) -> pd.DataFrame:
    """One row per threshold, sorted by pct_strong ascending (all-low ... all-high)."""
    p = np.asarray(p, dtype=float)
    thresholds = np.unique(np.concatenate([[-1.0, 1.0], np.quantile(p, np.linspace(0.0, 1.0, n_points))]))
    rows = [metrics_at(p, y, t, cost_ratio, scores) for t in thresholds]
    df = pd.DataFrame(rows).sort_values(["pct_strong", "threshold"], ascending=[True, False]).reset_index(drop=True)
    return df


def _auc(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    return float(np.sum(np.diff(x) * (y[:-1] + y[1:]) / 2.0))


def curve_summary(curve: pd.DataFrame) -> dict:
    """Area-style summaries (higher is better).
    apgr      = area under PGR-vs-%strong. Random routing scores ~0.5; a perfect router
                scores 1 - hard_share/2 (it reaches PGR=1 as soon as it has sent every
                hard prompt to the strong tier).
    lift_area = area between the quality curve and the random-routing line."""
    return {
        "apgr": _auc(curve["pct_strong"], curve["pgr"]),
        "lift_area": _auc(curve["pct_strong"], curve["quality"] - curve["random_quality"]),
    }


def pct_strong_for_quality(curve: pd.DataFrame, target: float) -> float:
    ok = curve[curve["quality"] >= target]
    return float(ok["pct_strong"].min()) if len(ok) else float("nan")


def random_pct_strong_for_quality(q_low: float, target: float) -> float:
    """Share of strong calls a RANDOM router needs to reach `target` quality."""
    if q_low >= target:
        return 0.0
    return float(min(1.0, (target - q_low) / (1.0 - q_low)))


def pick_threshold(curve: pd.DataFrame, target_quality: float, margin: float = 0.0) -> float:
    """Cheapest operating point whose quality >= target_quality + margin.
    If the target is unreachable, fall back to the highest-quality point."""
    ok = curve[curve["quality"] >= target_quality + margin]
    if ok.empty:
        best = curve["quality"].max()
        ok = curve[curve["quality"] >= best]
    row = ok.sort_values(["pct_strong", "threshold"], ascending=[True, False]).iloc[0]
    return float(row["threshold"])


def threshold_for_pct_strong(p, pct: float) -> float:
    """Threshold that sends (approximately) `pct` of the prompts to the strong tier."""
    p = np.asarray(p, dtype=float)
    if pct <= 0:
        return 1.0
    if pct >= 1:
        return -1.0
    return float(np.quantile(p, 1.0 - pct))
