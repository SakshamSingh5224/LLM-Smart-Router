"""Synthetic sample generator used for fast offline unit testing."""
from __future__ import annotations

import pandas as pd


def generate_synthetic_data(n: int = 100) -> pd.DataFrame:
    prompts = []
    tiers = []
    scores = []

    for i in range(n):
        if i % 2 == 0:
            prompts.append(f"Write a Python function to compute factorial {i}")
            tiers.append("high")
            scores.append(2)
        else:
            prompts.append(f"What is the capital of country {i}?")
            tiers.append("low")
            scores.append(5)

    return pd.DataFrame({"prompt": prompts, "tier": tiers, "mixtral_score": scores})
