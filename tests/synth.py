"""Tiny synthetic dataset shaped like data/processed/*.parquet (for offline tests)."""
import numpy as np
import pandas as pd

EASY = ["hi there", "what is the capital of {}?", "thanks a lot", "translate hello into {}", "give me a synonym for happy",
        "who wrote the book {}?", "what color is the sky", "how are you today"]
HARD = ["write a python function that {} and prove it terminates ```def f(x): return x```",
        "derive the equation for {} and solve it step by step with a proof",
        "design a distributed architecture for {} and analyze the trade-offs in detail with a table",
        "debug this recursive parser implementation and refactor it to be iterative {}"]
WORDS = ["france", "spanish", "sorting", "graphs", "payments", "caching", "queues", "dune", "rome", "linux"]


def make_df(n: int, seed: int = 0, noise: float = 0.1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        hard = rng.random() < 0.35
        tpl = rng.choice(HARD if hard else EASY)
        text = tpl.format(rng.choice(WORDS)) if "{}" in tpl else tpl
        if rng.random() < noise:
            hard = not hard  # label noise
        score = int(rng.choice([1, 2, 3]) if hard else rng.choice([4, 5, 5]))
        rows.append({"id": f"x-{i}", "prompt": text, "source": "synthetic", "mixtral_score": score,
                     "complexity": "complex" if hard else "simple", "tier": "high" if hard else "low"})
    return pd.DataFrame(rows)
