"""Gate 4: answer a fixed sample of questions end to end, for checking BY HAND.

Sample (chosen by rule, not by looking at answers): every third development
question (q01, q04, ..., q28), plus two questions the corpus cannot answer, which
should come back as NOT_IN_CORPUS. Dense retrieval, k from config.yaml.

For each answer it records whether the gold passage was among the excerpts sent,
so a wrong answer can be blamed on retrieval or on generation. Correctness itself
is left blank for a human to fill in: no LLM-as-judge (see CLAUDE.md).

Writes:  results/answer_check.json, results/answer_check.md
Takes:   ~45 s per question, ~9 minutes in total.

Usage:  python scripts/07_answer_check.py
"""

from __future__ import annotations

import json
import statistics
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from aeration_qa.config import load_config  # noqa: E402
from aeration_qa.embed import Embedder  # noqa: E402
from aeration_qa.generate import OllamaClient  # noqa: E402
from aeration_qa.index import load_index  # noqa: E402
from aeration_qa.pipeline import LLMStrategy, Pipeline, load_years  # noqa: E402
from aeration_qa.retrieve import DenseRetriever  # noqa: E402

# Real prefill on this CPU is ~30-50 tok/s. Faster means Ollama replayed a cached
# prompt prefix (the system prompt and any excerpt shared with the previous question;
# 2026-10-27 runs at 63-75 tok/s were partial hits), so the time is not real.
CACHED_PREFILL_TOK_S = 55

OUT_OF_CORPUS = [
    {"id": "x01", "question": "How does chlorine dose affect trihalomethane formation in drinking water?"},
    {"id": "x02", "question": "What does a UV disinfection system cost to install per cubic metre of capacity?"},
]


def main() -> int:
    cfg = load_config()
    dev = yaml.safe_load((ROOT / "data" / "questions_dev.yaml").read_text(encoding="utf-8"))["questions"]
    index = load_index(cfg.cache / "chunks.jsonl", cfg.cache / "index")
    sample = dev[::3] + OUT_OF_CORPUS
    for q in sample:
        q["gold"] = sorted(c["chunk_id"] for c in index.chunks if "snippets" in q
                           and c["paper_id"] == q["paper_id"] and any(s in c["text"] for s in q["snippets"]))

    embedder = Embedder(cfg.embedding_model, cfg.embedding_cache, cfg.query_prefix)
    pipeline = Pipeline(DenseRetriever(index.chunk_ids, index.embeddings, embedder),
                        LLMStrategy(OllamaClient(cfg.generator)), index.chunks, load_years(cfg.manifest),
                        cfg.retrieval.k)

    records = []
    for i, q in enumerate(sample, start=1):
        answer = pipeline.ask(q["question"])
        sent = [s.chunk_id for s in answer.sources]
        record = {
            "id": q["id"], "kind": q.get("kind", "out_of_corpus"), "question": q["question"],
            "answer": answer.text, "abstained": answer.abstained, "cited": answer.cited,
            "invalid_citations": answer.invalid_citations,
            "gold_sent": bool(set(q["gold"]) & set(sent)) if q["gold"] else None,
            "sources": [{"number": s.number, "chunk_id": s.chunk_id, "pages": s.pages, "section": s.section}
                        for s in answer.sources],
            **answer.stats,
        }
        record["prefill_cached"] = (record["prefill_seconds"] == 0
                                    or record["prompt_tokens"] / record["prefill_seconds"] > CACHED_PREFILL_TOK_S)
        records.append(record)
        print(f"[{i:>2}/{len(sample)}] {q['id']} {answer.stats['generation_seconds']:5.1f}s  "
              f"gold sent: {record['gold_sent']}  cited {answer.cited}  "
              f"{'ABSTAINED' if answer.abstained else ''}", flush=True)

    in_corpus = [r for r in records if r["kind"] != "out_of_corpus"]
    timed = [r for r in records if not r["prefill_cached"]]
    summary = {
        "date": str(date.today()), "model": cfg.generator.model, "retriever": "dense", "k": cfg.retrieval.k,
        "n": len(records), "n_timed": len(timed),
        "median_seconds": statistics.median(r["generation_seconds"] for r in timed),
        "max_seconds": max(r["generation_seconds"] for r in timed),
        "median_prompt_tokens": statistics.median(r["prompt_tokens"] for r in records),
        "gold_sent": sum(bool(r["gold_sent"]) for r in in_corpus), "n_in_corpus": len(in_corpus),
        "answered_with_citation": sum(bool(r["cited"]) and not r["abstained"] for r in records),
        "abstained_in_corpus": sum(r["abstained"] for r in in_corpus),
        "abstained_out_of_corpus": sum(r["abstained"] for r in records if r["kind"] == "out_of_corpus"),
        "invalid_citations": sum(len(r["invalid_citations"]) for r in records),
    }
    (cfg.results / "answer_check.json").write_text(
        json.dumps({"summary": summary, "answers": records}, indent=2), encoding="utf-8")
    (cfg.results / "answer_check.md").write_text(render(summary, records), encoding="utf-8")
    print("\n" + json.dumps(summary, indent=2))
    print(f"\nwritten to {cfg.results / 'answer_check.md'}")
    return 0


def render(s: dict, records: list[dict]) -> str:
    lines = [
        "# Gate 4: end-to-end answers, for checking by hand",
        "",
        f"Run {s['date']}: `{s['model']}`, {s['retriever']} retrieval, k={s['k']}. {s['n']} questions: every "
        "third development question plus two the papers cannot answer. Fill in the *Correct?* column by "
        "reading each answer against its cited pages; no automatic judge is used.",
        "",
        f"- Time per answer: median **{s['median_seconds']:.1f} s**, worst {s['max_seconds']:.1f} s "
        f"(median prompt {s['median_prompt_tokens']:.0f} tokens; {s['n'] - s['n_timed']} cache-served "
        "runs excluded)",
        f"- Correct passage among the excerpts sent: {s['gold_sent']} of {s['n_in_corpus']}",
        f"- Answers with at least one citation: {s['answered_with_citation']} of {s['n']}",
        f"- NOT_IN_CORPUS: {s['abstained_out_of_corpus']} of 2 unanswerable questions, "
        f"{s['abstained_in_corpus']} of {s['n_in_corpus']} answerable ones",
        f"- Citations to excerpts that were never sent: {s['invalid_citations']}",
        "",
        "| Question | Gold sent | Cited | Seconds | Correct? |",
        "|---|---|---|---|---|",
    ]
    for r in records:
        gold = {True: "yes", False: "**no**", None: "n/a"}[r["gold_sent"]]
        lines.append(f"| {r['id']} ({r['kind']}) | {gold} | {r['cited'] or '—'} | "
                     f"{r['generation_seconds']:.0f}{' (cached)' if r['prefill_cached'] else ''} | |")
    lines.append("")
    for r in records:
        lines += [f"## {r['id']}: {r['question']}", "", r["answer"], ""]
        lines += [f"- [{x['number']}] `{x['chunk_id']}` {x['pages']}, {x['section']}" for x in r["sources"]]
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
