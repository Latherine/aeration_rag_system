"""Prompt assembly, the Ollama call, and citation checking.

The model is never trusted with provenance. It is asked to cite `[n]`, but every
marker is checked against the blocks actually sent, and the Sources list shown to
the user is built by code from chunk metadata, not from anything the model wrote.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

import requests

from aeration_qa.config import GeneratorConfig

NOT_IN_CORPUS = "NOT_IN_CORPUS"

SYSTEM = (
    "You answer questions about wastewater treatment aeration control using only "
    "the provided excerpts. Cite every factual claim with a bracketed number, e.g. [2]. "
    f"If the excerpts do not contain the answer, reply exactly: {NOT_IN_CORPUS}. "
    "Do not use outside knowledge."
)

# [2], [1, 3], [1][2] and [1-3] are all citations; anything else in brackets is not.
_MARKER = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\]")
# The papers' own reference markers ("chloramines [1]") use the same syntax. Left in
# an excerpt, the model copies them and they pass as valid citations of OUR blocks.
_PAPER_REFERENCE = re.compile(r"\s*\[\d+(?:\s*[,–-]\s*\d+)*\]")


def strip_paper_references(text: str) -> str:
    return _PAPER_REFERENCE.sub("", text)


@dataclass(frozen=True)
class Source:
    """One numbered excerpt exactly as the generator saw it."""

    number: int
    retrieval_rank: int
    chunk_id: str
    title: str
    year: str
    section: str
    page_start: int
    page_end: int
    text: str

    @property
    def pages(self) -> str:
        return f"p.{self.page_start}" if self.page_start == self.page_end else f"pp.{self.page_start}-{self.page_end}"


def number_sources(chunks_best_first: list[dict], years: dict[str, str]) -> list[Source]:
    """Number excerpts in prompt order, with the highest-scoring chunk LAST.

    Small models over-attend to the end of the context, so the best evidence sits
    right before the question. [1] is therefore the weakest of the k excerpts.
    Text is stripped of the papers' own [n] markers; the retrieval index is not.
    """
    ordered = list(enumerate(chunks_best_first, start=1))[::-1]
    return [
        Source(number, rank, c["chunk_id"], c["title"], years.get(c["paper_id"], "n.d."), c["section"],
               c["page_start"], c["page_end"], strip_paper_references(c["text"]))
        for number, (rank, c) in enumerate(ordered, start=1)
    ]


def build_prompt(question: str, sources: list[Source]) -> str:
    blocks = [f"[{s.number}] {s.title} ({s.year}), {s.pages}\n{s.text}" for s in sources]
    return "\n\n".join(blocks) + f"\n\nQuestion: {question}"


def check_citations(text: str, n_sources: int) -> tuple[list[int], list[int]]:
    """Return (valid cited numbers, out-of-range numbers). Out-of-range = hallucination signal."""
    numbers: set[int] = set()
    for group in _MARKER.findall(text):
        for part in re.split(r"\s*,\s*", group):
            ends = [int(x) for x in re.split(r"\s*[–-]\s*", part)]
            numbers.update(range(ends[0], ends[-1] + 1) if len(ends) == 2 else ends)
    valid = sorted(n for n in numbers if 1 <= n <= n_sources)
    invalid = sorted(n for n in numbers if not 1 <= n <= n_sources)
    return valid, invalid


@dataclass(frozen=True)
class Generation:
    text: str
    prompt_tokens: int
    prefill_seconds: float
    output_tokens: int
    decode_seconds: float
    wall_seconds: float


class OllamaClient:
    def __init__(self, cfg: GeneratorConfig, model: str | None = None):
        self.cfg = cfg
        self.model = model or cfg.model

    def chat(self, system: str, user: str) -> Generation:
        started = time.perf_counter()
        try:
            response = requests.post(
                f"{self.cfg.host}/api/chat",
                json={
                    "model": self.model,
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                    "stream": False,
                    "keep_alive": self.cfg.keep_alive,
                    # num_ctx on every request: a silent server default of 2048 would
                    # truncate the excerpts and look like a retrieval failure.
                    "options": {"temperature": self.cfg.temperature, "seed": self.cfg.seed, "top_p": 1.0,
                                "num_ctx": self.cfg.num_ctx, "num_predict": self.cfg.num_predict},
                },
                timeout=900,
            )
        except requests.ConnectionError as exc:
            raise RuntimeError(f"Ollama is not answering at {self.cfg.host}. Start it with `ollama serve`.") from exc
        if response.status_code == 404:
            raise RuntimeError(
                f"Ollama does not have {self.model}. Run `ollama list`; if it is empty, the server is "
                "reading the wrong model folder (OLLAMA_MODELS should be D:\\ollama-models).")
        response.raise_for_status()
        body = response.json()
        ns = 1_000_000_000
        return Generation(
            text=body["message"]["content"].strip(),
            prompt_tokens=body.get("prompt_eval_count", 0),
            prefill_seconds=body.get("prompt_eval_duration", 0) / ns,
            output_tokens=body.get("eval_count", 0),
            decode_seconds=body.get("eval_duration", 0) / ns,
            wall_seconds=time.perf_counter() - started,
        )
