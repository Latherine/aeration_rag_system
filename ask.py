"""Ask the papers a question.

Usage:
  .venv\\Scripts\\python.exe ask.py "What does the ammonia valley in the pH profile indicate?"
  .venv\\Scripts\\python.exe ask.py "..." --retriever bm25      # bm25 | dense | hybrid
  .venv\\Scripts\\python.exe ask.py "..." --extractive          # quote sentences, no model
  .venv\\Scripts\\python.exe ask.py "..." --show-prompt         # print exactly what the model read

Order matters for RAM: the question is embedded first (fastembed, in this
process), then the local model generates (Ollama, a separate process). The two
never compute at the same time.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from aeration_qa.config import load_config  # noqa: E402
from aeration_qa.embed import Embedder  # noqa: E402
from aeration_qa.generate import OllamaClient, build_prompt  # noqa: E402
from aeration_qa.index import load_index  # noqa: E402
from aeration_qa.pipeline import ExtractiveStrategy, LLMStrategy, Pipeline, load_years  # noqa: E402
from aeration_qa.retrieve import BM25Retriever, DenseRetriever, HybridRetriever  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question")
    parser.add_argument("--retriever", choices=("dense", "bm25", "hybrid"), default="dense")
    parser.add_argument("--extractive", action="store_true", help="quote sentences instead of generating")
    parser.add_argument("--k", type=int, help="excerpts per prompt (default: config.yaml)")
    parser.add_argument("--show-prompt", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    index = load_index(cfg.cache / "chunks.jsonl", cfg.cache / "index")
    embedder = Embedder(cfg.embedding_model, cfg.embedding_cache, cfg.query_prefix)
    bm25 = BM25Retriever(index.chunk_ids, [c["text"] for c in index.chunks])
    dense = DenseRetriever(index.chunk_ids, index.embeddings, embedder)
    retriever = {"bm25": bm25, "dense": dense,
                 "hybrid": HybridRetriever([bm25, dense], cfg.retrieval.rrf_k, cfg.retrieval.fusion_pool)}[args.retriever]
    strategy = ExtractiveStrategy(embedder) if args.extractive else LLMStrategy(OllamaClient(cfg.generator))
    pipeline = Pipeline(retriever, strategy, index.chunks, load_years(cfg.manifest), args.k or cfg.retrieval.k)

    answer = pipeline.ask(args.question)
    if args.show_prompt:
        print("=" * 30 + " prompt " + "=" * 30)
        print(build_prompt(answer.question, answer.sources))
        print("=" * 68 + "\n")
    print(answer.render())
    s = answer.stats
    timing = f"\n[{answer.retriever} retrieval {answer.retrieval_ms:.0f} ms | {answer.strategy}"
    if "prompt_tokens" in s:
        timing += (f" | read {s['prompt_tokens']} tokens in {s['prefill_seconds']:.1f} s, "
                   f"wrote {s['output_tokens']} in {s['decode_seconds']:.1f} s")
    print(timing + f" | total {s.get('generation_seconds', 0):.1f} s]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
