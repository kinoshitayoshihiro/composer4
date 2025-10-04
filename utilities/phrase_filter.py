from __future__ import annotations

from collections.abc import Sequence
from typing import List
try:
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:  # pragma: no cover - optional dependency
    CountVectorizer = None  # type: ignore
    cosine_similarity = None  # type: ignore
import warnings
try:  # optional dependency
    import hdbscan
except ImportError:  # pragma: no cover - optional
    hdbscan = None  # type: ignore


def _phrase_str(events: Sequence[dict]) -> str:
    return " ".join(f"{ev.get('instrument','x')}_{round(ev.get('offset',0)*100)}" for ev in events)


def cluster_phrases(events_list: Sequence[Sequence[dict]], n: int = 4) -> List[bool]:
    """Cluster phrases by 3-gram similarity and return keep mask."""
    if not events_list:
        return []
    texts = [_phrase_str(ev) for ev in events_list]
    if CountVectorizer is None or cosine_similarity is None:
        warnings.warn(
            "sklearn not installed; using naive dedupe", RuntimeWarning
        )
        return [True for _ in texts]
    vec = CountVectorizer(analyzer="word", ngram_range=(3, 3))
    X = vec.fit_transform(texts)
    sim = cosine_similarity(X)
    if hdbscan is None:
        warnings.warn(
            "hdbscan not installed; using Jaccard dedupe fallback",
            RuntimeWarning,
        )
        token_sets = [set(t.split()) for t in texts]

        def _jacc(a: set[str], b: set[str]) -> float:
            if not a and not b:
                return 1.0
            return len(a & b) / len(a | b)

        keep: List[bool] = []
        kept_sets: list[set[str]] = []
        for ts in token_sets:
            if any(_jacc(ts, ks) >= 0.75 for ks in kept_sets):
                keep.append(False)
            else:
                keep.append(True)
                kept_sets.append(ts)
        while len(keep) < len(texts):
            keep.append(True)
        return keep
    clusterer = hdbscan.HDBSCAN(min_cluster_size=n, metric="precomputed")
    labels = clusterer.fit_predict(1 - sim)
    keep: List[bool] = []
    seen: set[int] = set()
    for lab in labels:
        if lab == -1 or lab not in seen:
            keep.append(True)
            if lab != -1:
                seen.add(lab)
        else:
            keep.append(False)
    return keep

__all__ = ["cluster_phrases"]
