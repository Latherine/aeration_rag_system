"""BGE-small embeddings via fastembed (ONNX, no torch).

BGE is trained asymmetrically: questions carry an instruction prefix, passages
do not. In fastembed 0.8.1, `query_embed()` does NOT add that prefix for BGE —
it returns exactly what `embed()` returns (verified 2026-10-13). So the prefix
is added here, explicitly. Getting this wrong raises no error; it just quietly
makes dense retrieval worse, which is why tests/test_embed_prefix.py exists.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding


class Embedder:
    def __init__(self, model_name: str, cache_dir: Path, query_prefix: str):
        self._model = TextEmbedding(model_name=model_name, cache_dir=str(cache_dir))
        self.query_prefix = query_prefix

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        return _normalize(np.array(list(self._model.embed(list(texts))), dtype=np.float32))

    def embed_query(self, question: str) -> np.ndarray:
        vector = np.array(next(iter(self._model.embed([self.query_prefix + question]))), dtype=np.float32)
        return vector / np.linalg.norm(vector)


def _normalize(matrix: np.ndarray) -> np.ndarray:
    """Unit-length rows, so cosine similarity is a plain dot product."""
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
