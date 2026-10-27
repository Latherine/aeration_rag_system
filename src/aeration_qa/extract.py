"""PDF -> clean, reading-ordered paragraphs, one record per text block.

Academic papers are two-column, carry running heads and page numbers, and
use typographic ligatures and soft hyphens. Each of those silently corrupts
retrieval if left in: interleaved columns produce sentences that never
existed, ligatures make "eﬄuent" a different BM25 term from "effluent", and
the references section floods term statistics with author names.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import fitz

TOP_MARGIN = 0.075
BOTTOM_MARGIN = 0.93
RIGHT_EDGE = 0.95
MIN_WORDS = 6

_HEADING = re.compile(
    r"^(?:\d+(?:\.\d+)*\.?|[IVX]+\.|[A-Z]\.)\s+[A-Z]"
    r"|^(?:Abstract|Introduction|Conclusions?|Discussion|Results|Methods?)\b",
    re.IGNORECASE,
)
_REFERENCES = re.compile(r"^\s*references\s*$", re.IGNORECASE)
# Publisher metadata: searchable it only produces matches that answer nothing.
_BOILERPLATE = re.compile(
    r"^(?:https?://|doi:|received\b|accepted\b|available online|copyright\b|©|\(c\)|"
    r"contents lists available|journal homepage|corresponding author|e-?mail addresses?\b|"
    r"academic editor\b|correspondence should be addressed)",
    re.IGNORECASE,
)
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


@dataclass(frozen=True)
class Block:
    paper_id: str
    page: int
    section: str
    text: str


def clean_text(raw: str) -> str:
    """Normalize one block: ligatures, soft hyphens, line-break hyphens, whitespace."""
    text = unicodedata.normalize("NFKC", raw)
    text = _CONTROL.sub("", text)
    text = text.replace("\xad\n", "").replace("\xad", "")
    # A hyphen at a line end followed by a lowercase letter is a word broken
    # across lines; keep real compounds like "pre-\nAnoxic" or "NH4-\nN" intact.
    text = re.sub(r"(\w)-\n(?=[a-z])", r"\1", text)
    text = re.sub(r"-\n\s*", "-", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    return re.sub(r"[ \t\xa0]+", " ", text).strip()


def is_heading(text: str) -> bool:
    return len(text.split()) <= 16 and bool(_HEADING.match(text))


def order_blocks(blocks: list[tuple], page_width: float) -> list[tuple]:
    """Reading order for a two-column page.

    Full-width blocks split the page into horizontal bands; inside each band
    the left column is read top to bottom, then the right column.
    """
    mid = page_width / 2
    full = sorted((b for b in blocks if b[0] < mid - 40 and b[2] > mid + 40), key=lambda b: b[1])
    columns = [b for b in blocks if b not in full]

    ordered: list[tuple] = []
    band_top = float("-inf")
    for divider in [*full, None]:
        band_bottom = divider[1] if divider else float("inf")
        band = [b for b in columns if band_top <= b[1] < band_bottom]
        ordered += sorted((b for b in band if b[0] < mid), key=lambda b: b[1])
        ordered += sorted((b for b in band if b[0] >= mid), key=lambda b: b[1])
        if divider:
            ordered.append(divider)
            band_top = divider[1]
    return ordered


def extract_pdf(path: Path, paper_id: str) -> list[Block]:
    records: list[Block] = []
    section = "front matter"
    with fitz.open(path) as doc:
        for page_number, page in enumerate(doc, start=1):
            width, height = page.rect.width, page.rect.height
            raw = [
                b for b in page.get_text("blocks")
                if b[6] == 0
                and b[3] > TOP_MARGIN * height
                and b[1] < BOTTOM_MARGIN * height
                and b[0] < RIGHT_EDGE * width
            ]
            for block in order_blocks(raw, width):
                text = clean_text(block[4])
                if not text:
                    continue
                if _REFERENCES.match(text):
                    return records
                if is_heading(text):
                    section = text
                    continue
                if len(text.split()) < MIN_WORDS or _BOILERPLATE.match(text):
                    continue
                records.append(Block(paper_id, page_number, section, text))
    return records
