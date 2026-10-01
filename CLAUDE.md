# Aeration QA — Project Instructions

A **retrieval-augmented question-answering assistant for wastewater-treatment aeration control**.
Ask it an operator/engineer question in plain language; it retrieves the relevant passages from a small
corpus of real papers and answers with citations back to paper, page and section.

This is a personal research/portfolio project. It is **not** a course deliverable and is not tied to
any other repo's conventions.

## Why this project exists

It grew out of a literature review of four families of aeration-control approaches: fixed-schedule
optimization via process simulation; real-time phase control from sensor "bending points" (the ammonia
valley in pH, the DO elbow, the nitrate knee and nitrate apex in ORP); model-based DO trajectory
optimization; and ML-based aeration control. Aeration electricity is reported to be more than half of a
treatment plant's running cost, so the domain matters — but the deliverable here is a *language* system
over that literature, not a control system.

**Important framing:** an earlier attempt at this idea (`sbr-transition-transformer/`, in the user's
separate `water-quality-forecasting` repo) built a transformer that localizes phase transitions in
synthetic pH/ORP/DO curves. It is **paused, not abandoned** — but it is a numeric time-series model that
merely *uses* the transformer architecture. It is not a language model. This project is the genuine
language-model project.

## Mandatory rules (do not violate)

These are hardware and licensing constraints, not preferences. Each one has cost someone a wasted
afternoon at least once, so do not re-litigate them without new evidence.

- **CPU only. 7.7 GB total RAM, of which ~3–4 GB is actually usable** (Windows takes the rest).
  The machine is an i5-1235U with Intel Iris Xe integrated graphics — there is no CUDA and no usable
  GPU acceleration.
- **Never propose fine-tuning or training a model.** On this hardware it is not slow, it is infeasible.
  The generator is always an off-the-shelf quantized open-weight model.
- **Generator is capped at 1.5B–3B parameters, Q4 quantized.** A 7B model at Q4 is ~4.5 GB and will
  page to disk, at which point throughput collapses by an order of magnitude and the symptom looks like
  "the model got slow" rather than "out of RAM".
- **Never use a reasoning/"thinking" model.** At 6–10 tokens/sec a 1,500-token reasoning trace is three
  minutes of wall clock before the answer even starts. Use explicitly non-thinking `-Instruct` variants.
- **No paid API, ever** — no OpenAI, no Anthropic, no hosted embedding service. Everything runs locally
  on open weights. This is a hard requirement from the user, not a cost optimization.
- **Never add `torch` or `sentence-transformers`.** They would pull a ~2.5 GB wheel and ~1 GB of
  resident memory at inference, which is fatal on a 3–4 GB budget shared with the LLM. `fastembed`
  (ONNX) does the same job in ~200 MB resident.
- **Never add langchain or llama-index.** They would hide the retrieval mechanics that this project
  exists to measure. Building retrieval by hand *is* the project.
- **Embedding and generation must never run at the same time.** Ollama is a separate resident process;
  the ONNX embedding session is in ours. Index-building embeds and exits; the interactive path embeds
  one query vector, then generates. Set `OLLAMA_MAX_LOADED_MODELS=1`.
- **Always set `num_ctx` explicitly on every Ollama request.** Server defaults have changed across
  versions; silently getting 2048 truncates the context and presents as a retrieval failure.
- **BGE embeddings need a query prefix on queries but not on passages.** fastembed applies it in
  `query_embed()` and *not* in `embed()`. Using `embed()` for both silently destroys dense retrieval
  quality — this is the single most common way people cripple BGE. `tests/test_embed_prefix.py` guards
  this; do not delete that test.
- **PyMuPDF is AGPL-3.0.** This was a conscious, accepted tradeoff (it is the only extractor that gives
  block bounding boxes and font sizes, which are required to reconstruct two-column academic papers).
  The README must state it. If this repo is ever made public, the licence implication is real.
- **Never commit the PDFs** in `data/papers/` — they are copyrighted. `data/MANIFEST.csv` records
  provenance (including a sha256 per file) so someone else can obtain the same corpus.

## Evaluation philosophy

Inherited deliberately from the user's `water-quality-forecasting` repo, where a trivial persistence
baseline beat every ML model in 7 of 9 cases and an LSTM was deleted from the pipeline for not earning
its complexity. The same skepticism applies here:

- **BM25 keyword search is a strong baseline, not a strawman.** On a ~10-paper corpus dense in rare
  jargon (*ammonia valley*, *nitrate knee*, *DO elbow*) there is no vocabulary mismatch for embeddings
  to bridge, and BM25 may well win outright. "BM25 beat the neural embeddings" is a legitimate,
  publishable headline — write it up as a finding, not a failure.
- **Report paired bootstrap confidence intervals, not point estimates.** A positive difference whose CI
  crosses zero is noise.
- **LLM-as-judge is refused, on purpose.** The only judge available locally is the same weak model being
  evaluated, and a model judging its own output has a documented self-preference bias. Declining a
  metric with a stated reason is a stronger methodological signal than producing numbers that look
  rigorous and mean nothing. If you add automatic answer metrics, label them honestly as proxies.

## Stack, and what was rejected

| Component | Choice | Rejected alternative and why |
|---|---|---|
| Generator | Ollama + `qwen2.5:3b-instruct-q4_K_M` | `llama-cpp-python` — prebuilt wheels lag new CPython minors; source build needs MSVC + CMake |
| Embeddings | `fastembed` + BGE-small-en-v1.5 | `sentence-transformers` — drags in torch (RAM) |
| Vector store | numpy brute-force cosine | FAISS — wheel risk for zero benefit at ~400 chunks (a 0.6 MB matrix) |
| Keyword search | `rank-bm25` | `bm25s` is ~100× faster and you will never notice at this scale |
| PDF extraction | PyMuPDF | `pypdf` returns interleaved garbage on two-column papers |
| HTTP to Ollama | `requests` | the `ollama` client pulls httpx + pydantic (Rust extension = wheel risk) |

Environment: **Windows 11, Python 3.13**. PyMuPDF ships a Stable-ABI wheel covering 3.10–3.14 and
fastembed/onnxruntime both have 3.13 wheels, so there is no known wheel gap. If fastembed ever breaks,
pin `onnxruntime==1.22.*` first.

## Design decisions worth knowing before changing code

- **Chunks are ~300 tokens with 60-token overlap, sentence-aligned.** BGE-small silently truncates at
  512 tokens, so an oversized chunk loses its tail from the embedding while BM25 still sees all of it —
  which would quietly bias the central BM25-vs-dense comparison. 300 also keeps prefill latency
  tolerable: four chunks ≈ 1,400 context tokens ≈ 15–30 s of prefill on this CPU *before the first
  output token*.
- **Each chunk stores `text` and `embed_text` separately.** Dense retrieval embeds
  `"{title} | {section}\n\n{text}"`; BM25 indexes bare `text`. Prepending titles to the BM25 index would
  inflate each title's term frequency across every chunk of that paper. This asymmetry is deliberate.
- **The references section is cut during extraction.** It is ~15% of a paper and pure author-name and
  journal-title noise that otherwise dominates BM25 term statistics.
- **Citations are never trusted to the model.** A Sources block is always appended from chunk metadata
  by code, so provenance survives even if the model cites nothing. Inline `[n]` markers are validated
  against the blocks actually sent; out-of-range markers are counted as a hallucination signal.
- **`pipeline.py` treats generation as a swappable strategy.** If the local model turns out to be too
  slow on this CPU, the fallback is extractive QA — return the 2–3 most query-relevant sentences
  verbatim with citations. That is fast, cannot hallucinate, and leaves the retrieval layer (where this
  project's value lives) completely intact. Do not hard-wire generation into the pipeline.

## Status

**Phase 0 complete (2026-09-29). Gate 0 passed at k=3.**

Built and verified: Python 3.13 venv with every dependency resolving to a working wheel (and `torch`
confirmed absent), Ollama 0.35.0 with `OLLAMA_MODELS` redirected to `D:\ollama-models`,
`qwen2.5:3b-instruct-q4_K_M` pulled (1.84 GB). `scripts/00_check_env.py` and `scripts/00_smoke_test.py`
both pass; measurements are in `results/smoke_test.json`.

### What the benchmark found

| Passages per prompt | Prompt tokens | End to end |
|---|---|---|
| 4 (the original design) | 1,398 | 48.3 s |
| **3 (chosen)** | **1,081** | **43.5 s** |
| 2 | 762 | 30.3 s |

Measured throughput: **prefill ~31-38 tok/s, decode ~8-10 tok/s.**

**Prefill is ~80% of the cost.** The model spends ~8 s writing an answer and ~35 s merely *reading* the
retrieved passages. This is why `k` was reduced from 4 to 3 rather than dropping to a smaller model:
the bottleneck is prompt length, not model size. Anyone tempted to "fix" latency by shrinking the model
should re-read that sentence first.

**Honest caveat on the margin.** 43.5 s clears the 45 s gate, but run-to-run variance is roughly 15%
(k=4 measured 56.1 s in one session and 48.3 s in another; decode rate swung between 4 and 10 tok/s,
presumably thermal or background load). k=3 is a *marginal* pass, not a comfortable one. If latency
becomes painful in practice, k=2 has real headroom — and `k` lives in `config.yaml` precisely so this
stays a measured tradeoff rather than a constant someone guessed.

### Benchmark pitfall — do not reintroduce

The first version of `00_smoke_test.py` sent an **identical prompt** on every run and reported
**11,540 tok/s** prefill, yielding a confident and completely wrong "PASS". Ollama was replaying the
cached KV state of the prompt prefix rather than computing anything. Every benchmark here must vary the
prompt between runs — `build_prompt()` injects a unique marker at the head for exactly this reason.
Real questions retrieve different passages, so prefill genuinely runs every time.

### Next

1. Collect 10-12 papers into `data/papers/`, fill `data/MANIFEST.csv`, write `data/glossary.yaml`.
2. Ingestion (`extract.py`, `chunking.py`), then retrieval (`embed.py`, `index.py`, `retrieve.py`),
   then generation (`generate.py`, `pipeline.py`, `ask.py`).
3. The evaluation harness is **deliberately deferred** — a working pipeline comes first, then its scope
   gets sized against something real. Note that at ~43 s per answer, a ~300-call evaluation sweep is a
   ~3.5 hour overnight run; the retrieval half needs no model at all and runs in seconds.

The full technical plan lives in this repo at [docs/PLAN.md](docs/PLAN.md) — it covers the annotation
protocol and pre-registration discipline for phase 5, and a phase 6 extension that folds the paused
bending-point detector back in as a second information source (with the rule that application code,
never the model, decides when to run it — small local models are unreliable at function calling).
