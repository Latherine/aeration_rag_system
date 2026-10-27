"""Extracted blocks -> retrieval passages (chunks) of ~300 BGE tokens.

Rules, each with a reason:
- Measure length with BGE's own tokenizer. Technical text with units and
  numbers runs ~2 tokens per word, so a word-count estimate undercounts exactly
  the passages most likely to exceed BGE's silent 512-token truncation.
- Cut only at sentence boundaries, so no chunk starts or ends mid-sentence.
- Never cross a section boundary: a chunk should be about one thing.
- Overlap ~60 tokens (whole sentences) so a fact on a boundary appears whole
  in at least one chunk.
"""

from __future__ import annotations

import glob
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path

from tokenizers import Tokenizer

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[\"“])")
_ABBREVIATIONS = (
    "e.g.", "i.e.", "et al.", "Fig.", "Figs.", "Eq.", "Eqs.", "vs.", "approx.",
    "No.", "Ref.", "cf.", "Vol.", "pp.", "p.", "ca.", "resp.",
)
_CAPTION = re.compile(r"^(?:Fig\.|Figure|Table|Supplementary Table)\s*\d", re.IGNORECASE)
MIN_TAIL_TOKENS = 80
MIN_ASIDE_TOKENS = 40
NUMERIC_TABLE_RATIO = 0.35
BGE_MAX_TOKENS = 512


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    paper_id: str
    title: str
    section: str
    page_start: int
    page_end: int
    text: str
    embed_text: str
    n_tokens: int


@dataclass(frozen=True)
class _Sentence:
    text: str
    page: int
    tokens: int


def load_bge_tokenizer(cache_dir: Path) -> Tokenizer:
    matches = glob.glob(str(cache_dir / "**" / "tokenizer.json"), recursive=True)
    if not matches:
        raise FileNotFoundError(
            f"No BGE tokenizer under {cache_dir}. Run scripts/00_check_env.py once to download the model."
        )
    return Tokenizer.from_file(matches[0])


def token_counter(tokenizer: Tokenizer) -> Callable[[str], int]:
    return lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids)


def split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for part in _SENTENCE_END.split(text):
        if sentences and sentences[-1].endswith(_ABBREVIATIONS):
            sentences[-1] += " " + part
        else:
            sentences.append(part)
    return [s for s in sentences if s.strip()]


def _split_overlong(sentence: _Sentence, max_tokens: int, count: Callable[[str], int]) -> list[_Sentence]:
    """A 'sentence' longer than a whole chunk is almost always a flattened table; cut it by words."""
    pieces, words = [], []
    for word in sentence.text.split():
        if words and count(" ".join([*words, word])) > max_tokens:
            pieces.append(" ".join(words))
            words = []
        words.append(word)
    if words:
        pieces.append(" ".join(words))
    return [_Sentence(p, sentence.page, count(p)) for p in pieces]


def _pack(
    sentences: list[_Sentence], target: int, overlap: int, max_tokens: int
) -> list[list[_Sentence]]:
    groups: list[list[_Sentence]] = []
    current: list[_Sentence] = []
    size = 0
    for sentence in sentences:
        if current and size + sentence.tokens > target:
            groups.append(current)
            carry: list[_Sentence] = []
            carried = 0
            for previous in reversed(current):
                if carried >= overlap:
                    break
                carry.insert(0, previous)
                carried += previous.tokens
            # An overlap that is most of the previous chunk, or that would push
            # the next chunk past the hard cap, duplicates instead of bridging.
            if len(carry) == len(current) or carried > 2 * overlap or carried + sentence.tokens > max_tokens:
                carry, carried = [], 0
            current, size = carry, carried
        current.append(sentence)
        size += sentence.tokens
    if current:
        groups.append(current)

    # Fold a short tail into the previous chunk instead of leaving a fragment.
    if len(groups) > 1:
        tail = sum(s.tokens for s in groups[-1])
        before = sum(s.tokens for s in groups[-2])
        if tail < MIN_TAIL_TOKENS and before + tail <= max_tokens:
            seen = {id(s) for s in groups[-2]}
            groups[-2] += [s for s in groups[-1] if id(s) not in seen]
            groups.pop()
    return groups


def is_aside(text: str) -> bool:
    """Tables and figure captions: real content, but they interrupt sentences when inline."""
    if _CAPTION.match(text):
        return True
    words = text.split()
    numeric = sum(any(ch.isdigit() for ch in w) for w in words)
    return len(words) >= 8 and numeric / len(words) >= NUMERIC_TABLE_RATIO


def _merge_continuations(blocks: list[dict]) -> list[dict]:
    """A block starting lowercase continues the previous one across a column or page break."""
    merged: list[dict] = []
    for block in blocks:
        if merged and block["text"][:1].islower():
            merged[-1] = {**merged[-1], "text": merged[-1]["text"] + " " + block["text"]}
        else:
            merged.append(block)
    return merged


def _sentences(blocks: list[dict], cap: int, count: Callable[[str], int]) -> list[_Sentence]:
    sentences: list[_Sentence] = []
    for block in blocks:
        for text in split_sentences(block["text"]):
            sentence = _Sentence(text, block["page"], count(text))
            sentences += _split_overlong(sentence, cap, count) if sentence.tokens > cap else [sentence]
    return sentences


def chunk_paper(
    blocks: Iterable[dict],
    paper_id: str,
    title: str,
    count: Callable[[str], int],
    target: int,
    overlap: int,
    max_tokens: int,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section, section_blocks in groupby(blocks, key=lambda b: b["section"]):
        section_blocks = list(section_blocks)
        # The embedded text is "title | section" + passage, and BGE truncates the
        # whole thing at 512, so the passage budget shrinks by the header's size.
        header = f"{title} | {section}\n\n"
        cap = min(max_tokens, BGE_MAX_TOKENS - 2 - count(header))

        prose = _merge_continuations([b for b in section_blocks if not is_aside(b["text"])])
        asides = [b for b in section_blocks if is_aside(b["text"])]

        for stream, label in ((prose, section), (asides, f"{section} [tables/figures]")):
            sentences = _sentences(stream, cap, count)
            for group in _pack(sentences, min(target, cap), overlap, cap):
                text = " ".join(s.text for s in group)
                # A lone caption or bare table header matches queries but answers
                # nothing, since the figure itself is not in the text.
                if stream is asides and count(text) < MIN_ASIDE_TOKENS:
                    continue
                chunks.append(
                    Chunk(
                        chunk_id=f"{paper_id}:{len(chunks):03d}",
                        paper_id=paper_id,
                        title=title,
                        section=label,
                        page_start=min(s.page for s in group),
                        page_end=max(s.page for s in group),
                        text=text,
                        embed_text=f"{title} | {section}\n\n{text}",
                        n_tokens=count(text),
                    )
                )
    return chunks
