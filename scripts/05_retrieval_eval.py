"""Compare BM25, dense and hybrid retrieval on the development question set.

Reads data/questions_dev.yaml (answer keys fixed before this script first ran),
searches with all three methods, and writes:
  results/retrieval_eval.json     every number, per question and overall
  results/retrieval_eval.md       readable report with the charts embedded
  results/figures/*.png           two first-glance comparison charts

Usage:  python scripts/05_retrieval_eval.py
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from datetime import date
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from aeration_qa import figures  # noqa: E402
from aeration_qa.config import load_config  # noqa: E402
from aeration_qa.embed import Embedder  # noqa: E402
from aeration_qa.index import load_index  # noqa: E402
from aeration_qa.metrics import hit_at_k, mrr, paired_bootstrap  # noqa: E402
from aeration_qa.retrieve import BM25Retriever, DenseRetriever, HybridRetriever  # noqa: E402

DEPTH = 10
KINDS = ("shared", "abbreviation", "paraphrase")


def main() -> int:
    cfg = load_config()
    questions = yaml.safe_load((ROOT / "data" / "questions_dev.yaml").read_text(encoding="utf-8"))["questions"]
    index = load_index(cfg.cache / "chunks.jsonl", cfg.cache / "index")
    chunks = index.by_id()
    ids = index.chunk_ids

    for q in questions:
        q["gold"] = sorted(cid for cid, c in chunks.items()
                           if c["paper_id"] == q["paper_id"] and any(s in c["text"] for s in q["snippets"]))
        if not q["gold"]:
            raise SystemExit(f"{q['id']}: answer key matches no chunk: {q['snippets']}")

    embedder = Embedder(cfg.embedding_model, cfg.embedding_cache, cfg.query_prefix)
    bm25 = BM25Retriever(ids, [c["text"] for c in index.chunks])
    dense = DenseRetriever(ids, index.embeddings, embedder)
    hybrid = HybridRetriever([bm25, dense], cfg.retrieval.rrf_k, cfg.retrieval.fusion_pool)
    retrievers = {"bm25": bm25, "dense": dense, "hybrid": hybrid}

    ranks: dict[str, list[int | None]] = {m: [] for m in retrievers}
    latency_ms: dict[str, list[float]] = {m: [] for m in retrievers}
    for q in questions:
        gold = set(q["gold"])
        q["ranks"] = {}
        for name, retriever in retrievers.items():
            started = time.perf_counter()
            hits = retriever.search(q["question"], DEPTH)
            latency_ms[name].append((time.perf_counter() - started) * 1000)
            rank = next((h.rank for h in hits if h.chunk_id in gold), None)
            ranks[name].append(rank)
            q["ranks"][name] = rank

    k = cfg.retrieval.k
    summary = {
        name: {
            "hit_at_1": hit_at_k(r, 1), f"hit_at_{k}": hit_at_k(r, k), "hit_at_10": hit_at_k(r, 10),
            "mrr_at_10": mrr(r), "median_latency_ms": statistics.median(latency_ms[name]),
        }
        for name, r in ranks.items()
    }
    by_kind = {
        name: {kind: hit_at_k([r for r, q in zip(ranks[name], questions) if q["kind"] == kind], k)
               for kind in KINDS}
        for name in retrievers
    }
    counts = {kind: sum(q["kind"] == kind for q in questions) for kind in KINDS}
    comparisons = [
        ("Dense − BM25", paired_bootstrap(ranks["dense"], ranks["bm25"])),
        ("Hybrid − BM25", paired_bootstrap(ranks["hybrid"], ranks["bm25"])),
        ("Dense − Hybrid", paired_bootstrap(ranks["dense"], ranks["hybrid"])),
    ]

    fig_dir = cfg.results / "figures"
    for stale in fig_dir.glob("*.png"):
        stale.unlink()  # charts from earlier versions must not linger next to the current ones
    found = {m: sum(1 for x in r if x is not None and x <= k) for m, r in ranks.items()}
    paths = [
        figures.overall(found, len(questions), k, fig_dir / "1_overall_top3.png"),
        figures.by_kind(by_kind, counts, fig_dir / "2_top3_by_question_wording.png"),
    ]

    payload = {
        "date": str(date.today()), "n_questions": len(questions), "n_chunks": len(ids), "k": k,
        "summary": summary, "hit_at_k_by_kind": by_kind, "kind_counts": counts,
        "bootstrap_mrr_difference": {label: r for label, r in comparisons},
        "questions": [{key: q[key] for key in ("id", "kind", "question", "gold", "ranks")} for q in questions],
    }
    (cfg.results / "retrieval_eval.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (cfg.results / "retrieval_eval.md").write_text(
        render(payload, comparisons, [p.relative_to(cfg.results) for p in paths]), encoding="utf-8")

    print(f"{'method':<8}{'hit@1':>8}{'hit@' + str(k):>8}{'hit@10':>8}{'MRR@10':>8}{'ms':>7}")
    for name, s in summary.items():
        print(f"{name:<8}{s['hit_at_1']:>8.0%}{s[f'hit_at_{k}']:>8.0%}{s['hit_at_10']:>8.0%}"
              f"{s['mrr_at_10']:>8.3f}{s['median_latency_ms']:>7.1f}")
    print(f"\ntop-{k} hit rate by question type")
    for name in retrievers:
        print(f"  {name:<8}" + "  ".join(f"{kind} {by_kind[name][kind]:.0%}" for kind in KINDS))
    print("\nMRR difference, 95% paired bootstrap")
    for label, r in comparisons:
        print(f"  {label:<16}{r['diff']:+.3f}  [{r['low']:+.3f}, {r['high']:+.3f}]")
    print(f"\nwritten to {cfg.results / 'retrieval_eval.md'} and {fig_dir}")
    return 0


def render(payload: dict, comparisons: list, figure_paths: list[Path]) -> str:
    k, n = payload["k"], payload["n_questions"]
    lines = [
        "# Retrieval comparison: BM25 vs dense vs hybrid",
        "",
        f"Run {payload['date']} on {payload['n_chunks']} chunks from 3 papers, {n} development questions "
        f"({payload['kind_counts']['shared']} shared wording, {payload['kind_counts']['abbreviation']} "
        f"abbreviation, {payload['kind_counts']['paraphrase']} paraphrase). Answer keys were fixed in "
        "`data/questions_dev.yaml` before the first run. The developer wrote both the system and the "
        "questions, so this guides decisions; it is not the independent Phase 5 evaluation.",
        "",
        f"| Method | Top-1 | Top-{k} | Top-10 | MRR@10 | Median search time |",
        "|---|---|---|---|---|---|",
    ]
    for name, s in payload["summary"].items():
        lines.append(f"| {name} | {s['hit_at_1']:.0%} | {s[f'hit_at_{k}']:.0%} | {s['hit_at_10']:.0%} | "
                     f"{s['mrr_at_10']:.3f} | {s['median_latency_ms']:.1f} ms |")
    lines += ["", "| MRR difference | Estimate | 95% interval |", "|---|---|---|"]
    for label, r in comparisons:
        lines.append(f"| {label} | {r['diff']:+.3f} | [{r['low']:+.3f}, {r['high']:+.3f}] |")
    lines.append("")
    for path in figure_paths:
        lines += [f"![{path.stem}]({path.as_posix()})", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
