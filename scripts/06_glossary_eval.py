"""Does glossary query expansion help? Each method with and without data/glossary.yaml.

Same 28 development questions and answer keys as 05_retrieval_eval.py. The glossary
was written from the papers by the rule stated at its top, and fixed before this
script first ran. Caveat: the "abbreviation" question type exists because Gate 3
showed BM25 missing abbreviations, so these questions were written with this
experiment's weakness in mind. Treat a gain as a hypothesis for Phase 5, not a result.

Writes:
  results/glossary_eval.json          every number, per question and overall
  results/glossary_eval.md            readable report
  results/figures/3_glossary_effect.png

Usage:  python scripts/06_glossary_eval.py
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from aeration_qa import figures  # noqa: E402
from aeration_qa.config import load_config  # noqa: E402
from aeration_qa.embed import Embedder  # noqa: E402
from aeration_qa.glossary import ExpandedRetriever, Glossary  # noqa: E402
from aeration_qa.index import load_index  # noqa: E402
from aeration_qa.metrics import hit_at_k, mrr, paired_bootstrap  # noqa: E402
from aeration_qa.retrieve import BM25Retriever, DenseRetriever, HybridRetriever  # noqa: E402

DEPTH = 10
KINDS = ("shared", "abbreviation", "paraphrase")
METHODS = ("bm25", "dense", "hybrid")


def main() -> int:
    cfg = load_config()
    questions = yaml.safe_load((ROOT / "data" / "questions_dev.yaml").read_text(encoding="utf-8"))["questions"]
    index = load_index(cfg.cache / "chunks.jsonl", cfg.cache / "index")
    chunks = index.by_id()
    ids = index.chunk_ids
    glossary = Glossary.load(cfg.glossary)

    for q in questions:
        q["gold"] = sorted(cid for cid, c in chunks.items()
                           if c["paper_id"] == q["paper_id"] and any(s in c["text"] for s in q["snippets"]))
        if not q["gold"]:
            raise SystemExit(f"{q['id']}: answer key matches no chunk: {q['snippets']}")
        q["expanded"] = glossary.expand(q["question"])

    embedder = Embedder(cfg.embedding_model, cfg.embedding_cache, cfg.query_prefix)
    bm25 = BM25Retriever(ids, [c["text"] for c in index.chunks])
    dense = DenseRetriever(ids, index.embeddings, embedder)
    plain = {"bm25": bm25, "dense": dense,
             "hybrid": HybridRetriever([bm25, dense], cfg.retrieval.rrf_k, cfg.retrieval.fusion_pool)}
    # Expanding the hybrid's query expands what both of its retrievers see.
    expanded = {m: ExpandedRetriever(r, glossary) for m, r in plain.items()}

    ranks: dict[str, dict[str, list[int | None]]] = {"plain": {}, "glossary": {}}
    for condition, retrievers in (("plain", plain), ("glossary", expanded)):
        for name, retriever in retrievers.items():
            ranks[condition][name] = []
            for q in questions:
                gold = set(q["gold"])
                rank = next((h.rank for h in retriever.search(q["question"], DEPTH) if h.chunk_id in gold), None)
                ranks[condition][name].append(rank)
                q.setdefault("ranks", {})[f"{name}+glossary" if condition == "glossary" else name] = rank

    k = cfg.retrieval.k
    summary = {
        cond: {m: {f"hit_at_{k}": hit_at_k(r, k), "hit_at_10": hit_at_k(r, 10), "mrr_at_10": mrr(r),
                   "by_kind": {kind: hit_at_k([x for x, q in zip(r, questions) if q["kind"] == kind], k)
                               for kind in KINDS}}
               for m, r in by_method.items()}
        for cond, by_method in ranks.items()
    }
    bootstrap = {m: paired_bootstrap(ranks["glossary"][m], ranks["plain"][m]) for m in METHODS}
    counts = {kind: sum(q["kind"] == kind for q in questions) for kind in KINDS}
    n_expanded = sum(q["expanded"] != q["question"] for q in questions)

    found = {cond: {m: sum(1 for x in r if x is not None and x <= k) for m, r in by_method.items()}
             for cond, by_method in ranks.items()}
    chart = figures.glossary_effect(found["plain"], found["glossary"], len(questions), k,
                                    cfg.results / "figures" / "3_glossary_effect.png")

    payload = {
        "date": str(date.today()), "n_questions": len(questions), "n_chunks": len(ids), "k": k,
        "n_glossary_terms": len(glossary.terms), "n_questions_expanded": n_expanded, "kind_counts": counts,
        "summary": summary, "bootstrap_mrr_glossary_minus_plain": bootstrap,
        "questions": [{key: q[key] for key in ("id", "kind", "question", "expanded", "gold", "ranks")}
                      for q in questions],
    }
    (cfg.results / "glossary_eval.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (cfg.results / "glossary_eval.md").write_text(
        render(payload, chart.relative_to(cfg.results)), encoding="utf-8")

    print(f"{n_expanded} of {len(questions)} questions changed by the glossary ({len(glossary.terms)} terms)\n")
    print(f"{'method':<16}{'top-' + str(k):>7}{'MRR@10':>8}   " + "  ".join(f"{kind:>12}" for kind in KINDS))
    for m in METHODS:
        for cond in ("plain", "glossary"):
            s = summary[cond][m]
            label = m if cond == "plain" else f"{m}+glossary"
            print(f"{label:<16}{found[cond][m]:>4}/{len(questions)}{s['mrr_at_10']:>8.3f}   "
                  + "  ".join(f"{s['by_kind'][kind]:>12.0%}" for kind in KINDS))
    print("\nMRR difference (glossary - plain), 95% paired bootstrap")
    for m, r in bootstrap.items():
        print(f"  {m:<8}{r['diff']:+.3f}  [{r['low']:+.3f}, {r['high']:+.3f}]")
    print(f"\nwritten to {cfg.results / 'glossary_eval.md'}")
    return 0


def render(payload: dict, chart: Path) -> str:
    k, n = payload["k"], payload["n_questions"]
    s = payload["summary"]
    lines = [
        "# Glossary query expansion: does it help?",
        "",
        f"Run {payload['date']} on {payload['n_chunks']} chunks, the same {n} development questions as "
        f"`retrieval_eval.md`. The glossary (`data/glossary.yaml`, {payload['n_glossary_terms']} terms) was "
        "written from the papers by a fixed rule before this first run; it changed "
        f"{payload['n_questions_expanded']} of the {n} questions. The abbreviation question type was written "
        "after Gate 3 exposed BM25's abbreviation misses, so a gain here is a hypothesis to confirm in "
        "Phase 5, not a result.",
        "",
        f"| Method | Top-{k} | Top-10 | MRR@10 | Top-{k}: own words | abbreviation | everyday words |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in METHODS:
        for cond in ("plain", "glossary"):
            r = s[cond][m]
            label = m if cond == "plain" else f"{m} + glossary"
            lines.append(f"| {label} | {r[f'hit_at_{k}']:.0%} | {r['hit_at_10']:.0%} | {r['mrr_at_10']:.3f} | "
                         + " | ".join(f"{r['by_kind'][kind]:.0%}" for kind in KINDS) + " |")
    lines += ["", "| MRR difference (glossary − plain) | Estimate | 95% interval |", "|---|---|---|"]
    for m, r in payload["bootstrap_mrr_glossary_minus_plain"].items():
        lines.append(f"| {m} | {r['diff']:+.3f} | [{r['low']:+.3f}, {r['high']:+.3f}] |")
    lines += ["", f"![{chart.stem}]({chart.as_posix()})", "", "## Questions the glossary changed", "",
              "| Question | Added | BM25 rank (before → after) |", "|---|---|---|"]
    for q in payload["questions"]:
        if q["expanded"] != q["question"]:
            added = q["expanded"][len(q["question"]):].strip()
            before, after = q["ranks"]["bm25"], q["ranks"]["bm25+glossary"]
            lines.append(f"| {q['id']} ({q['kind']}) | {added} | {before or '—'} → {after or '—'} |")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
