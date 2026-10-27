"""ask(question) -> Answer: retrieve, number the excerpts, answer, attach sources.

How the answer text is produced is a swappable strategy:
  LLMStrategy         the local model writes an answer citing [n]
  ExtractiveStrategy  the most question-relevant sentences, verbatim, each cited
The extractive one is the fallback if generation is too slow on this CPU. It
cannot hallucinate, and the retrieval layer above it is identical either way.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from aeration_qa.chunking import split_sentences
from aeration_qa.embed import Embedder
from aeration_qa.generate import NOT_IN_CORPUS, SYSTEM, OllamaClient, Source, build_prompt, check_citations, \
    number_sources
from aeration_qa.retrieve import Retriever


@dataclass(frozen=True)
class Draft:
    text: str
    stats: dict = field(default_factory=dict)


class Strategy(Protocol):
    name: str

    def answer(self, question: str, sources: list[Source]) -> Draft: ...


class LLMStrategy:
    def __init__(self, client: OllamaClient):
        self.client = client
        self.name = f"llm:{client.model}"

    def answer(self, question: str, sources: list[Source]) -> Draft:
        g = self.client.chat(SYSTEM, build_prompt(question, sources))
        return Draft(g.text, {
            "prompt_tokens": g.prompt_tokens, "prefill_seconds": round(g.prefill_seconds, 2),
            "output_tokens": g.output_tokens, "decode_seconds": round(g.decode_seconds, 2),
            "generation_seconds": round(g.wall_seconds, 2),
        })


class ExtractiveStrategy:
    """Pick the n sentences closest in meaning to the question and quote them."""

    name = "extractive"
    MIN_WORDS = 6  # shorter "sentences" are headings and table fragments

    def __init__(self, embedder: Embedder, n_sentences: int = 3):
        self.embedder = embedder
        self.n_sentences = n_sentences

    def answer(self, question: str, sources: list[Source]) -> Draft:
        started = time.perf_counter()
        # Neighbouring chunks overlap by ~60 tokens, so one sentence can arrive twice.
        # Keep the copy from the best-ranked excerpt (highest number = best).
        unique: dict[str, int] = {}
        for s in sorted(sources, key=lambda s: -s.number):
            for sentence in split_sentences(s.text):
                if len(sentence.split()) >= self.MIN_WORDS:
                    unique.setdefault(sentence, s.number)
        candidates = [(number, sentence) for sentence, number in unique.items()]
        if not candidates:
            return Draft(NOT_IN_CORPUS)
        scores = self.embedder.embed_passages([c[1] for c in candidates]) @ self.embedder.embed_query(question)
        best = np.argsort(-scores, kind="stable")[: self.n_sentences]
        text = "\n".join(f"- \"{candidates[i][1]}\" [{candidates[i][0]}]" for i in best)
        return Draft(text, {"generation_seconds": round(time.perf_counter() - started, 2)})


@dataclass(frozen=True)
class Answer:
    question: str
    text: str
    sources: list[Source]
    cited: list[int]
    invalid_citations: list[int]
    abstained: bool
    retriever: str
    strategy: str
    retrieval_ms: float
    stats: dict

    def render(self) -> str:
        lines = [self.text, "", "Sources (attached by code, not written by the model):"]
        for s in self.sources:
            mark = "*" if s.number in self.cited else " "
            lines.append(f" {mark}[{s.number}] {s.title} ({s.year}), {s.pages}, {s.section}  <{s.chunk_id}>")
        lines.append("   * = cited in the answer")
        if self.invalid_citations:
            lines.append(f"WARNING: the answer cites {self.invalid_citations}, which were not among the excerpts sent.")
        return "\n".join(lines)


def load_years(manifest: Path) -> dict[str, str]:
    with manifest.open(encoding="utf-8", newline="") as f:
        return {row["paper_id"]: row["year"] for row in csv.DictReader(f)}


class Pipeline:
    def __init__(self, retriever: Retriever, strategy: Strategy, chunks: list[dict], years: dict[str, str], k: int):
        self.retriever = retriever
        self.strategy = strategy
        self.by_id = {c["chunk_id"]: c for c in chunks}
        self.years = years
        self.k = k

    def ask(self, question: str) -> Answer:
        started = time.perf_counter()
        hits = self.retriever.search(question, self.k)
        retrieval_ms = (time.perf_counter() - started) * 1000
        sources = number_sources([self.by_id[h.chunk_id] for h in hits], self.years)
        draft = self.strategy.answer(question, sources)
        cited, invalid = check_citations(draft.text, len(sources))
        return Answer(question, draft.text, sources, cited, invalid,
                      abstained=draft.text.strip().startswith(NOT_IN_CORPUS),
                      retriever=self.retriever.name, strategy=self.strategy.name,
                      retrieval_ms=round(retrieval_ms, 1), stats=draft.stats)
