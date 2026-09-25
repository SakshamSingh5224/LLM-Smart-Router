"""Turn prompts into feature matrices for the classifier.

* TfidfFeaturizer      - word 1-2gram TF-IDF + hand-crafted features (sparse, ~1 ms/prompt)
* EmbeddingFeaturizer  - BGE-small sentence embeddings via ONNX (fastembed, no PyTorch)
                         + hand-crafted features (dense, ~10-30 ms/prompt on CPU)
"""
from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from .features import handcrafted_matrix


def _fingerprint(texts: Sequence[str]) -> str:
    h = hashlib.md5(str(len(texts)).encode())
    for t in texts:
        h.update(t[:64].encode("utf-8", "ignore"))
        h.update(b"\x00")
    return h.hexdigest()


class TfidfFeaturizer:
    kind = "tfidf"

    def __init__(self, max_features: int = 150_000, min_df: int = 3, max_chars: int = 4000):
        self.max_features, self.min_df, self.max_chars = max_features, min_df, max_chars
        self.vec: Optional[TfidfVectorizer] = None
        self.scaler: Optional[StandardScaler] = None

    def _clip(self, texts):
        return [(t or "")[: self.max_chars] for t in texts]

    def fit(self, texts: Sequence[str]) -> "TfidfFeaturizer":
        self.vec = TfidfVectorizer(
            ngram_range=(1, 2), min_df=self.min_df, max_features=self.max_features,
            sublinear_tf=True, dtype=np.float32,
        )
        self.vec.fit(self._clip(texts))
        self.scaler = StandardScaler().fit(handcrafted_matrix(texts))
        return self

    def transform(self, texts: Sequence[str], cache_path: Optional[str] = None):
        xt = self.vec.transform(self._clip(texts))
        xh = self.scaler.transform(handcrafted_matrix(texts)).astype(np.float32)
        return sp.hstack([xt, sp.csr_matrix(xh)], format="csr")

    def describe(self) -> dict:
        return {"kind": self.kind, "vocab": len(self.vec.vocabulary_)}


class EmbeddingFeaturizer:
    kind = "embed"

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", max_chars: int = 1200, batch_size: int = 128):
        self.model_name, self.max_chars, self.batch_size = model_name, max_chars, batch_size
        self.scaler: Optional[StandardScaler] = None
        self._model = None  # lazy, never pickled

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_model"] = None
        return state

    def _get_model(self):
        if self._model is None:
            from fastembed import TextEmbedding  # imported lazily: optional dependency

            self._model = TextEmbedding(model_name=self.model_name)
        return self._model

    def _embed(self, texts: Sequence[str], verbose: bool = False) -> np.ndarray:
        model = self._get_model()
        clipped = [(t or "")[: self.max_chars] or " " for t in texts]
        out, chunk, t0 = [], 2048, time.time()
        for i in range(0, len(clipped), chunk):
            out.extend(model.embed(clipped[i : i + chunk], batch_size=self.batch_size))
            if verbose:
                done = min(i + chunk, len(clipped))
                rate = done / max(time.time() - t0, 1e-9)
                print(f"\r    embedding {done}/{len(clipped)}  ({rate:.0f} prompts/s)", end="", file=sys.stderr, flush=True)
        if verbose:
            print(file=sys.stderr)
        return np.asarray(out, dtype=np.float32)

    def fit(self, texts: Sequence[str]) -> "EmbeddingFeaturizer":
        self.scaler = StandardScaler().fit(handcrafted_matrix(texts))
        return self

    def transform(self, texts: Sequence[str], cache_path: Optional[str] = None) -> np.ndarray:
        emb = None
        fp = _fingerprint(texts) if cache_path else None
        if cache_path and Path(cache_path).exists():
            z = np.load(cache_path, allow_pickle=False)
            if str(z["fp"]) == fp:
                emb = z["emb"]
        if emb is None:
            emb = self._embed(texts, verbose=len(texts) > 500)
            if cache_path:
                Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
                np.savez(cache_path, emb=emb, fp=np.array(fp))
        xh = self.scaler.transform(handcrafted_matrix(texts)).astype(np.float32)
        return np.hstack([emb, xh])

    def describe(self) -> dict:
        return {"kind": self.kind, "model": self.model_name}


def make_featurizer(kind: str, **opts):
    if kind == "tfidf":
        return TfidfFeaturizer(**opts)
    if kind == "embed":
        return EmbeddingFeaturizer(**opts)
    raise ValueError(f"unknown featurizer kind: {kind}")
