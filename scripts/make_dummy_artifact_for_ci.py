#!/usr/bin/env python3
"""Train a tiny router artifact on synthetic data - CI-only, for validating that
`docker build` succeeds (the Dockerfiles COPY results/router_artifact.joblib).
Not meant for real routing quality; `make train-router` on the real dataset
produces the artifact you actually deploy.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

import joblib  # noqa: E402
from synth import make_df  # noqa: E402

from smartrouter.training import train_router  # noqa: E402

train_df, val_df = make_df(400, seed=1), make_df(150, seed=2)
artifact, _ = train_router(train_df, val_df, backends=("tfidf",), Cs=(1.0,),
                            featurizer_opts={"tfidf": {"min_df": 1}}, log=lambda *_: None)

out = Path("results/router_artifact.joblib")
out.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(artifact, out, compress=3)
print(f"Wrote CI placeholder artifact to {out}")
