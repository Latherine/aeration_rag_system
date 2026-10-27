"""Gate 3 — do the three retrievers find passages we know are right?

Six hand-written questions, each with an answer key fixed BEFORE running any
search: a (paper_id, snippet) anchor, so the key survives re-chunking. A
question counts as found if any chunk containing its snippet ranks in the top
k. Five questions reuse the papers' vocabulary; the sixth deliberately does not
("more air than they need" vs. the paper's "over-aerated"), the case where
meaning-based search should beat keyword search.

Answer-key change, disclosed: the first question originally accepted only the
chunk defining the ammonia valley (p.2). Reading the first run's results showed
a p.5 chunk also answers it ("ammonia valley ... indicate the completion of a
nitrification process"), so that snippet was added AFTER seeing results. The
Phase 5 evaluation must not do this; its key is fixed before any run.

This is a smoke test with six questions, not the evaluation: no statistical
claim can be made from it. Its job is to show the pipeline works end to end and
to surface obvious failures before the real labelled question set is built.

Usage:  python scripts/04_retrieval_check.py
Writes: results/retrieval_check.md and results/retrieval_check.json
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from aeration_qa.config import load_config  # noqa: E402
from aeration_qa.embed import Embedder  # noqa: E402
from aeration_qa.index import load_index  # noqa: E402
from aeration_qa.retrieve import BM25Retriever, DenseRetriever, HybridRetriever  # noqa: E402

QUESTIONS = [
    ("What does the ammonia valley in the pH profile indicate?",
     "abuhasan2010", ("which indicate the end of nitrification",
                      "especially ammonia valley"), "shared wording"),
    ("How many hours of aeration were needed before it could be stopped in the low-strength biofilter test?",
     "abuhasan2010", ("aeration can be stopped at 7 h HRT",), "shared wording"),
    ("How much money per year did optimizing aeration time save across Duhok's SBR plants?",
     "hasan2026", ("234,049,308",), "shared wording"),
    ("What solids retention time is recommended for stable nitrification?",
     "sakthivel2025", ("Target an SRT of approximately",), "shared wording"),
    ("Which reactor configuration gave the highest total nitrogen and phosphorus removal?",
     "sakthivel2025", ("The highest simultaneous TN and TP removal occurred",), "shared wording"),
    ("Are the plants pumping more air into the tanks than they actually need?",
     "hasan2026", ("over-aerated",), "paraphrase: no shared key term"),
]
DEPTH = 10


def main() -> int:
    cfg = load_config()
    index = load_index(cfg.cache / "chunks.jsonl", cfg.cache / "index")
    chunks = index.by_id()
    ids = index.chunk_ids
    embedder = Embedder(cfg.embedding_model, cfg.embedding_cache, cfg.query_prefix)

    bm25 = BM25Retriever(ids, [c["text"] for c in index.chunks])
    dense = DenseRetriever(ids, index.embeddings, embedder)
    hybrid = HybridRetriever([bm25, dense], cfg.retrieval.rrf_k, cfg.retrieval.fusion_pool)
    retrievers = [bm25, dense, hybrid]
    k = cfg.retrieval.k

    records = []
    for question, paper_id, snippets, kind in QUESTIONS:
        gold = {cid for cid, c in chunks.items()
                if c["paper_id"] == paper_id and any(s in c["text"] for s in snippets)}
        assert gold, f"answer key snippets not found in any chunk: {snippets!r}"
        record = {"question": question, "kind": kind, "gold": sorted(gold), "methods": {}}
        for retriever in retrievers:
            hits = retriever.search(question, DEPTH)
            first = next((h.rank for h in hits if h.chunk_id in gold), None)
            record["methods"][retriever.name] = {
                "first_relevant_rank": first,
                "found_in_top_k": first is not None and first <= k,
                "top": [{"chunk_id": h.chunk_id, "score": round(h.score, 4)} for h in hits[:k]],
            }
        records.append(record)

    summary = {}
    for retriever in retrievers:
        ranks = [r["methods"][retriever.name]["first_relevant_rank"] for r in records]
        summary[retriever.name] = {
            "found_in_top_k": sum(1 for x in ranks if x is not None and x <= k),
            "mrr_at_10": round(sum(1 / x for x in ranks if x) / len(ranks), 3),
        }

    cfg.results.mkdir(parents=True, exist_ok=True)
    (cfg.results / "retrieval_check.json").write_text(
        json.dumps({"date": str(date.today()), "k": k, "n_chunks": len(ids), "summary": summary,
                    "questions": records}, indent=2), encoding="utf-8")
    (cfg.results / "retrieval_check.md").write_text(
        render(records, summary, k, len(ids), chunks), encoding="utf-8")

    print(f"{'method':<8}{'found in top ' + str(k):>16}{'MRR@10':>9}")
    for name, s in summary.items():
        print(f"{name:<8}{s['found_in_top_k']:>11} / {len(records)}{s['mrr_at_10']:>9}")
    print()
    for r in records:
        marks = "  ".join(
            f"{n}:{m['first_relevant_rank'] or '-':>2}" for n, m in r["methods"].items())
        print(f"{marks}   {r['question'][:70]}")
    print(f"\nrank of first correct passage (- = not in top {DEPTH})")
    print(f"written to {cfg.results / 'retrieval_check.md'}")
    return 0


def render(records: list[dict], summary: dict, k: int, n_chunks: int, chunks: dict) -> str:
    lines = [
        "# Retrieval check (Gate 3)",
        "",
        f"Run {date.today()} on {n_chunks} chunks from 3 papers. Six hand-written questions, answer key "
        "fixed before any search. A smoke test, **not** the evaluation: six questions support no "
        "statistical claim.",
        "",
        "**Disclosed change:** after the first run, the answer key for the ammonia-valley question was "
        "widened to also accept a p.5 passage that states the answer (\"ammonia valley ... indicate the "
        "completion of a nitrification process\"). First run: BM25 2/6, dense 4/6, hybrid 2/6.",
        "",
        f"| Method | Correct passage in top {k} | MRR@10 |",
        "|---|---|---|",
    ]
    for name, s in summary.items():
        lines.append(f"| {name} | {s['found_in_top_k']} / {len(records)} | {s['mrr_at_10']} |")
    lines += ["", "Rank of the first correct passage (— = not in top 10):", "",
              "| Question | Kind | BM25 | Dense | Hybrid |", "|---|---|---|---|---|"]
    for r in records:
        ranks = [r["methods"][m]["first_relevant_rank"] or "—" for m in ("bm25", "dense", "hybrid")]
        lines.append(f"| {r['question']} | {r['kind']} | {ranks[0]} | {ranks[1]} | {ranks[2]} |")
    lines += ["", "## Top passages per question", ""]
    for r in records:
        lines += [f"### {r['question']}", "", f"Correct: {', '.join(r['gold'])}", ""]
        for name, m in r["methods"].items():
            lines.append(f"**{name}**")
            lines.append("")
            for i, hit in enumerate(m["top"], start=1):
                c = chunks[hit["chunk_id"]]
                mark = " ✓" if hit["chunk_id"] in r["gold"] else ""
                pages = f"p.{c['page_start']}" + (f"–{c['page_end']}" if c["page_end"] != c["page_start"] else "")
                lines.append(f"{i}. `{hit['chunk_id']}`{mark} ({pages}, {c['section'][:50]}): "
                             f"{c['text'][:160]}…")
            lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
