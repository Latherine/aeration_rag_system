"""Build once, load many: chunk embeddings on disk plus the chunk records.

The index stores a fingerprint of chunks.jsonl. If the chunks change (a new
paper, a chunking fix) and the index is not rebuilt, loading refuses instead of
silently pairing old vectors with new text.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aeration_qa.embed import Embedder


@dataclass(frozen=True)
class Index:
    chunks: list[dict]
    embeddings: np.ndarray

    @property
    def chunk_ids(self) -> list[str]:
        return [c["chunk_id"] for c in self.chunks]

    def by_id(self) -> dict[str, dict]:
        return {c["chunk_id"]: c for c in self.chunks}


def _fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_index(chunks_path: Path, index_dir: Path, embedder: Embedder, model_name: str) -> Index:
    chunks = [json.loads(line) for line in chunks_path.open(encoding="utf-8")]
    # Dense retrieval embeds title + section + text; BM25 later indexes bare text.
    embeddings = embedder.embed_passages([c["embed_text"] for c in chunks])

    index_dir.mkdir(parents=True, exist_ok=True)
    np.save(index_dir / "embeddings.npy", embeddings)
    (index_dir / "meta.json").write_text(json.dumps({
        "chunks_sha256": _fingerprint(chunks_path),
        "model": model_name,
        "n_chunks": len(chunks),
        "dim": int(embeddings.shape[1]),
    }, indent=2), encoding="utf-8")
    return Index(chunks, embeddings)


def load_index(chunks_path: Path, index_dir: Path) -> Index:
    meta_path = index_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No index in {index_dir}. Run scripts/03_build_index.py first.")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta["chunks_sha256"] != _fingerprint(chunks_path):
        raise RuntimeError(
            f"Index in {index_dir} was built from a different {chunks_path.name}. "
            "Rerun scripts/03_build_index.py."
        )
    chunks = [json.loads(line) for line in chunks_path.open(encoding="utf-8")]
    return Index(chunks, np.load(index_dir / "embeddings.npy"))
