"""Phase 3 — embed every chunk once and save the index to cache/index/.

Run after 02_chunk.py, and again whenever the chunks change; searching refuses
a stale index. Embedding only — Ollama is not touched (RAM choreography rule).

Usage:  python scripts/03_build_index.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from aeration_qa.config import load_config  # noqa: E402
from aeration_qa.embed import Embedder  # noqa: E402
from aeration_qa.index import build_index  # noqa: E402


def main() -> int:
    cfg = load_config()
    embedder = Embedder(cfg.embedding_model, cfg.embedding_cache, cfg.query_prefix)
    started = time.perf_counter()
    index = build_index(cfg.cache / "chunks.jsonl", cfg.cache / "index", embedder, cfg.embedding_model)
    seconds = time.perf_counter() - started
    rows, dim = index.embeddings.shape
    size_kb = index.embeddings.nbytes / 1024
    print(f"embedded {rows} chunks -> {rows} x {dim} matrix ({size_kb:.0f} KB) in {seconds:.1f}s")
    print(f"written to {cfg.cache / 'index'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
