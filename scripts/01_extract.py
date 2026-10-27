"""Phase 2 — extract every paper in data/MANIFEST.csv to cache/extracted/.

Writes one JSONL file per paper (one record per paragraph block, with page and
section) plus a plain-text dump for reading by eye. Chunking into retrieval
passages is a later step; this stage only has to get the text right.

Usage:  python scripts/01_extract.py
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from aeration_qa.extract import extract_pdf  # noqa: E402

PAPERS = ROOT / "data" / "papers"
OUT = ROOT / "cache" / "extracted"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "data" / "MANIFEST.csv").open(encoding="utf-8") as f:
        manifest = list(csv.DictReader(f))

    print(f"{'paper':<16}{'pages':>6}{'blocks':>8}{'words':>8}  sections")
    for row in manifest:
        path = PAPERS / row["filename"]
        if not path.exists():
            print(f"{row['paper_id']:<16} MISSING {path.name}")
            continue
        blocks = extract_pdf(path, row["paper_id"])

        with (OUT / f"{row['paper_id']}.jsonl").open("w", encoding="utf-8") as f:
            for block in blocks:
                f.write(json.dumps(asdict(block), ensure_ascii=False) + "\n")

        with (OUT / f"{row['paper_id']}.txt").open("w", encoding="utf-8") as f:
            current = None
            for block in blocks:
                if block.section != current:
                    current = block.section
                    f.write(f"\n## [p{block.page}] {current}\n\n")
                f.write(block.text + "\n\n")

        words = sum(len(b.text.split()) for b in blocks)
        pages = len({b.page for b in blocks})
        sections = len({b.section for b in blocks})
        print(f"{row['paper_id']:<16}{pages:>6}{len(blocks):>8}{words:>8}  {sections}")

    print(f"\nwritten to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
