"""Glossary query expansion: add the other spelling of every abbreviation a query uses.

"What solids retention time is recommended?" becomes
"What solids retention time is recommended? SRT", so BM25 can match a paper that
only ever writes "SRT". Expansion only appends; the original words always stay.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Term:
    abbr: str
    long: tuple[str, ...]
    abbr_pattern: re.Pattern
    long_patterns: tuple[re.Pattern, ...]


def _long_pattern(phrase: str) -> re.Pattern:
    # Spaces and hyphens are interchangeable ("ion-selective" = "ion selective"); no
    # trailing \b, so the plural ("reactors") also matches.
    words = [re.escape(w) for w in re.split(r"[\s-]+", phrase)]
    return re.compile(r"\b" + r"[\s-]+".join(words), re.IGNORECASE)


class Glossary:
    def __init__(self, terms: list[dict]):
        self.terms = [
            Term(t["abbr"], tuple(t["long"]),
                 # Case-sensitive: "DO" is dissolved oxygen, "do" is a verb.
                 re.compile(rf"\b{re.escape(t['abbr'])}s?\b"),
                 tuple(_long_pattern(p) for p in t["long"]))
            for t in terms
        ]

    @classmethod
    def load(cls, path: Path) -> Glossary:
        return cls(yaml.safe_load(path.read_text(encoding="utf-8"))["terms"])

    def expand(self, query: str) -> str:
        added: list[str] = []
        for term in self.terms:
            has_abbr = bool(term.abbr_pattern.search(query))
            has_long = any(p.search(query) for p in term.long_patterns)
            if has_abbr and not has_long:
                added.extend(term.long)
            elif has_long and not has_abbr:
                added.append(term.abbr)
        return f"{query} {' '.join(added)}" if added else query


class ExpandedRetriever:
    """Wraps any retriever so its queries are glossary-expanded first."""

    def __init__(self, retriever, glossary: Glossary):
        self.name = f"{retriever.name}+glossary"
        self._retriever = retriever
        self._glossary = glossary

    def search(self, query: str, k: int):
        return self._retriever.search(self._glossary.expand(query), k)
