#!/usr/bin/env python3
"""Download RouteLLM's GPT-4-judged Arena dataset and build routing labels.

Source : routellm/gpt4_dataset  (train ~109k, validation ~10k)
Output : data/processed/{train,val,test}.parquet + stats.json

Columns kept: id, prompt, source, mixtral_score, complexity, tier
  tier == "low"  -> weak model was good enough (mixtral_score >= threshold)
  tier == "high" -> strong model needed

Splits (no prompt appears in more than one split):
  test  = the dataset's own `validation` split   (never used for training/calibration)
  train / val = 90 / 10 stratified split of the dataset's `train` split
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
from datasets import load_dataset  # noqa: E402

from smartrouter.config import load_settings  # noqa: E402
from smartrouter.labels import score_to_complexity, score_to_tier  # noqa: E402


def to_frame(split, threshold: int) -> pd.DataFrame:
    df = split.to_pandas()[["prompt", "source", "mixtral_score"]].copy()
    df = df.dropna(subset=["prompt", "mixtral_score"])
    df["prompt"] = df["prompt"].astype(str).str.strip()
    df = df[df["prompt"].str.len() > 0]
    # `source` is a list like ["lmsys-chat-1m"]; keep the first entry
    df["source"] = df["source"].apply(lambda s: s[0] if isinstance(s, (list, tuple)) and len(s) else str(s))
    df["mixtral_score"] = df["mixtral_score"].astype(int)
    df["complexity"] = df["mixtral_score"].apply(score_to_complexity)
    df["tier"] = df["mixtral_score"].apply(lambda s: score_to_tier(s, threshold))
    return df.drop_duplicates(subset="prompt").reset_index(drop=True)


def describe(df: pd.DataFrame) -> dict:
    return {
        "rows": int(len(df)),
        "tier_counts": df["tier"].value_counts().to_dict(),
        "tier_share": df["tier"].value_counts(normalize=True).round(4).to_dict(),
        "complexity_counts": df["complexity"].value_counts().to_dict(),
        "score_counts": {int(k): int(v) for k, v in df["mixtral_score"].value_counts().sort_index().items()},
        "top_sources": df["source"].value_counts().head(8).to_dict(),
        "prompt_chars": {k: round(float(v), 1) for k, v in df["prompt"].str.len().describe(percentiles=[.5, .95]).items()},
    }


def main() -> None:
    cfg = load_settings()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threshold", type=int, default=cfg.weak_score_threshold,
                    help="mixtral_score >= this => 'low' tier (default from .env, 4)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=cfg.data_dir / "processed")
    args = ap.parse_args()

    print(f"==> Downloading {cfg.hf_dataset} from Hugging Face (about 300 MB the first time)...")
    ds = load_dataset(cfg.hf_dataset)
    print("    splits:", {k: len(v) for k, v in ds.items()})

    full_train = to_frame(ds["train"], args.threshold)
    test = to_frame(ds["validation"], args.threshold)

    # avoid leakage: drop any test prompt that also appears in train
    test = test[~test["prompt"].isin(set(full_train["prompt"]))].reset_index(drop=True)

    val = full_train.groupby("tier", group_keys=False).sample(frac=0.10, random_state=args.seed)
    train = full_train.drop(val.index).reset_index(drop=True)
    val = val.reset_index(drop=True)

    for name, df in (("train", train), ("val", val), ("test", test)):
        df.insert(0, "id", [f"{name}-{i}" for i in range(len(df))])

    args.out.mkdir(parents=True, exist_ok=True)
    stats = {}
    for name, df in (("train", train), ("val", val), ("test", test)):
        df.to_parquet(args.out / f"{name}.parquet", index=False)
        stats[name] = describe(df)
        print(f"\n[{name}] rows={len(df):,}  tier_share={stats[name]['tier_share']}")

    stats["config"] = {"threshold": args.threshold, "seed": args.seed, "dataset": cfg.hf_dataset}
    (args.out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"\n==> Wrote parquet files + stats.json to {args.out}")
    major = max(stats["test"]["tier_share"].values())
    print(f"    Majority-class baseline on test = {major:.1%}. A router must beat this AND "
          f"correctly escalate the hard prompts.")


if __name__ == "__main__":
    main()
