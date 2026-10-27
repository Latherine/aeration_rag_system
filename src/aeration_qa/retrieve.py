"""Three retrieval conditions behind one interface: BM25, dense, and RRF hybrid.

Every condition returns the same `Hit` list, so evaluation code never needs to
know which one it is comparing. Nothing here is tuned: BM25 uses library
defaults, RRF uses the literature constant 60. Tuning on the same questions used
to compare the methods would leak the answer key into the comparison.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from rank_bm25 import BM25Okapi

from aeration_qa.embed import Embedder

_TOKEN = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")
# "do" is deliberately absent: lowercased, it is DO (dissolved oxygen), a core term here.
_STOPWORDS = frozenset(
    "a an and are as at be been by can could did does for from had has have how in into is it its "
    "may of on or should that the their then there these this to was were what when where which while "
    "who why will with would".split()
)


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    score: float
    rank: int


class Retriever(Protocol):
    name: str

    def search(self, query: str, k: int) -> list[Hit]: ...


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens that keep `nh4-n`, `kwh`, `2.5` intact. No stemming."""
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


def _top_k(scores: np.ndarray, chunk_ids: list[str], k: int) -> list[Hit]:
    order = np.argsort(-scores, kind="stable")[:k]
    return [Hit(chunk_ids[i], float(scores[i]), rank) for rank, i in enumerate(order, start=1)]


class BM25Retriever:
    name = "bm25"

    def __init__(self, chunk_ids: list[str], texts: list[str]):
        self._ids = chunk_ids
        self._bm25 = BM25Okapi([tokenize(t) for t in texts])

    def search(self, query: str, k: int) -> list[Hit]:
        return _top_k(np.asarray(self._bm25.get_scores(tokenize(query))), self._ids, k)


class DenseRetriever:
    name = "dense"

    def __init__(self, chunk_ids: list[str], embeddings: np.ndarray, embedder: Embedder):
        self._ids = chunk_ids
        self._matrix = embeddings
        self._embedder = embedder

    def search(self, query: str, k: int) -> list[Hit]:
        return _top_k(self._matrix @ self._embedder.embed_query(query), self._ids, k)


class HybridRetriever:
    """Reciprocal rank fusion: score = sum over methods of 1 / (rrf_k + rank).

    Fuses ranks, not scores, because BM25 scores are unbounded while cosine
    lives in [-1, 1]; any score normalization would be one more untested knob.
    """

    name = "hybrid"

    def __init__(self, retrievers: list[Retriever], rrf_k: int, pool: int):
        self._retrievers = retrievers
        self._rrf_k = rrf_k
        self._pool = pool

    def search(self, query: str, k: int) -> list[Hit]:
        fused: dict[str, float] = {}
        for retriever in self._retrievers:
            for hit in retriever.search(query, self._pool):
                fused[hit.chunk_id] = fused.get(hit.chunk_id, 0.0) + 1.0 / (self._rrf_k + hit.rank)
        ranked = sorted(fused.items(), key=lambda item: (-item[1], item[0]))[:k]
        return [Hit(chunk_id, score, rank) for rank, (chunk_id, score) in enumerate(ranked, start=1)]
