"""Feature extraction pipelines combining TF-IDF and handcrafted text signals."""
from __future__ import annotations
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from .features import extract_signals

class HandcraftedFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        for text in X:
            s = extract_signals(str(text))
            features.append([
                float(s["word_count"]) / 100.0,
                float(s["char_count"]) / 500.0,
                float(s["has_code"]),
                float(s["has_math"]),
                float(s["has_reasoning"]),
                float(s["is_long"]),
            ])
        return np.array(features)

def build_tfidf_pipeline() -> Pipeline:
    union = FeatureUnion([
        ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words="english")),
        ("handcrafted", HandcraftedFeatureExtractor()),
    ])
    return Pipeline([("features", union)])
