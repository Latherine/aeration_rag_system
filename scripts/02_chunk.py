"""Phase 2 — cut extracted papers into retrieval passages (chunks).

Reads cache/extracted/<paper_id>.jsonl (run 01_extract.py first), writes
cache/chunks.jsonl, and prints the checks Gate 2 asks for: size distribution,
anything over BGE's 512-token limit, chunks starting mid-sentence, and a few
random chunks to read by eye.

Usage:  python scripts/02_chunk.py [--samples N]
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from aeration_qa.chunking import chunk_paper, load_bge_tokenizer, token_counter  # noqa: E402
from aeration_qa.config import load_config  # noqa: E402

BGE_LIMIT = 512


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args()

    cfg = load_config()
    tokenizer = load_bge_tokenizer(cfg.embedding_cache)
    count = token_counter(tokenizer)

    with cfg.manifest.open(encoding="utf-8") as f:
        manifest = list(csv.DictReader(f))

    chunks = []
    for row in manifest:
        extracted = cfg.cache / "extracted" / f"{row['paper_id']}.jsonl"
        if not extracted.exists():
            print(f"skip {row['paper_id']}: no {extracted.name} (run 01_extract.py)")
            continue
        blocks = [json.loads(line) for line in extracted.open(encoding="utf-8")]
        chunks += chunk_paper(
            blocks, row["paper_id"], row["title"], count,
            target=cfg.chunking.target_tokens,
            overlap=cfg.chunking.overlap_tokens,
            max_tokens=cfg.chunking.max_tokens,
        )

    out = cfg.cache / "chunks.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")

    sizes = [c.n_tokens for c in chunks]
    embed_sizes = [len(tokenizer.encode(c.embed_text).ids) for c in chunks]  # includes [CLS]/[SEP]
    over_limit = [c.chunk_id for c, n in zip(chunks, embed_sizes) if n > BGE_LIMIT]
    mid_sentence = [c.chunk_id for c in chunks if c.text[:1].islower()]

    print(f"{'paper':<16}{'chunks':>7}")
    for paper_id in dict.fromkeys(c.paper_id for c in chunks):
        print(f"{paper_id:<16}{sum(c.paper_id == paper_id for c in chunks):>7}")
    print()
    print(f"total chunks          {len(chunks)}")
    print(f"tokens min/median/max {min(sizes)} / {statistics.median(sizes):.0f} / {max(sizes)}")
    print(f"embed_text max tokens {max(embed_sizes)}  (BGE limit {BGE_LIMIT})")
    print(f"over BGE limit        {len(over_limit)} {over_limit[:5]}")
    print(f"start mid-sentence    {len(mid_sentence)} {mid_sentence[:5]}")

    rng = random.Random(0)
    for chunk in rng.sample(chunks, min(args.samples, len(chunks))):
        pages = f"p.{chunk.page_start}" + (f"-{chunk.page_end}" if chunk.page_end != chunk.page_start else "")
        print("\n" + "-" * 78)
        print(f"{chunk.chunk_id}  {pages}  [{chunk.section}]  {chunk.n_tokens} tokens")
        print(chunk.text[:700] + (" ..." if len(chunk.text) > 700 else ""))

    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
