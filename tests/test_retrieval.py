import numpy as np
import pytest

from aeration_qa.config import load_config
from aeration_qa.embed import Embedder
from aeration_qa.retrieve import BM25Retriever, Hit, HybridRetriever, tokenize


@pytest.fixture(scope="module")
def embedder():
    cfg = load_config()
    return Embedder(cfg.embedding_model, cfg.embedding_cache, cfg.query_prefix)


def test_query_and_passage_embeddings_differ(embedder):
    """Guards the BGE prefix rule. fastembed 0.8.1's query_embed() does NOT add it,
    so if this ever passes trivially equal vectors, dense retrieval has silently degraded."""
    text = "What is the ammonia valley?"
    query = embedder.embed_query(text)
    passage = embedder.embed_passages([text])[0]
    assert not np.allclose(query, passage, atol=1e-4)
    assert np.isclose(np.linalg.norm(query), 1.0) and np.isclose(np.linalg.norm(passage), 1.0)


def test_tokenizer_keeps_domain_terms_whole():
    assert tokenize("NH4-N removal of 87.1 % at 4.5 kWh") == ["nh4-n", "removal", "87.1", "4.5", "kwh"]
    assert tokenize("The DO elbow") == ["do", "elbow"]  # DO = dissolved oxygen, never a stopword


def test_bm25_ranks_shared_term_first():
    bm25 = BM25Retriever(["a", "b", "c"], ["settling tank cleaning", "ammonia valley in pH", "blower energy use"])
    assert bm25.search("where is the ammonia valley", 1)[0].chunk_id == "b"


class _Fixed:
    def __init__(self, name, order):
        self.name, self._order = name, order

    def search(self, query, k):
        return [Hit(cid, 0.0, rank) for rank, cid in enumerate(self._order[:k], start=1)]


def test_rrf_prefers_chunk_both_methods_rank_highly():
    fused = HybridRetriever([_Fixed("x", ["c7", "c2", "c9"]), _Fixed("y", ["c2", "c5", "c7"])], rrf_k=60, pool=3)
    assert [h.chunk_id for h in fused.search("q", 3)] == ["c2", "c7", "c5"]
