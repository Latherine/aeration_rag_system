# Local RAG QA Assistant for Wastewater Aeration Control

## Context

An earlier direction (`sbr-transition-transformer/` — a transformer that localizes phase-transition
points in synthetic pH/ORP/DO curves) is **paused, not deleted**: its config, landmark primitives and
`SBRCycleSimulator` are already committed to `github.com/Latherine/water-quality-forecasting`. It was
paused because it is not actually a language-model project — it is a numeric time-series model that
happens to use the transformer architecture.

This project is a **question-answering assistant for wastewater-treatment aeration control**: a RAG
system over a small corpus of real papers, answering operator/engineer questions with citations. It
grows out of the same literature the user researched (fixed-schedule optimization; real-time phase
control from sensor bending points; model-based DO trajectory optimization; ML-based aeration control).

**Hard constraints, verified on this machine:** Windows 11, Python 3.13, CPU only (i5-1235U, no CUDA),
**7.7 GB total RAM** (~3–4 GB usable after Windows), no paid API of any kind, no LLM runtime installed
yet. Corpus is 5–15 PDFs the user collects. These rule out fine-tuning entirely and cap the generator
at a 1.5B–3B quantized model.

**The consequence that shapes everything:** a 3B model on this hardware writes mediocre prose. So the
project's value must live in **retrieval engineering and honest evaluation**, not in impressive
generated text — inheriting the culture of the user's other repo, where a trivial baseline beat ML in
7 of 9 cases and an LSTM was deleted for not earning its complexity.

Location: `D:\FALL 2026\COMP 560\aeration-qa` (new repo, `git init` locally; GitHub remote added later).

## Approach

### Stack

| Component | Choice | Why |
|---|---|---|
| Generator | **Ollama** + `qwen2.5:3b-instruct-q4_K_M` (~1.9 GB) | Lowest-friction Windows install, no admin/compiler. Fallback `qwen2.5:1.5b-instruct-q4_K_M`. **Never a "thinking"/reasoning model** — a 1,500-token reasoning trace at 6–10 tok/s is 3 minutes per answer. |
| Embeddings | **fastembed** + BGE-small-en-v1.5 | ONNX, ~200 MB resident. `sentence-transformers` would pull ~2.5 GB of torch and ~1 GB RSS — fatal on this RAM budget. |
| Vector store | **numpy brute-force cosine** | ~400 chunks × 384 dims = 0.6 MB. FAISS adds wheel risk for zero benefit. |
| Keyword baseline | **rank-bm25** | Pure Python, no wheel risk. No `nltk` (runtime corpus download). |
| PDF | **PyMuPDF** | Block bboxes + font sizes are needed for two-column reconstruction; `pypdf` returns interleaved garbage on two-column papers. **AGPL-3.0 — accepted; must be stated in the README.** |
| LLM call | **requests** | The `ollama` client pulls httpx+pydantic (Rust extension = wheel risk). The API is two JSON endpoints. |

Deliberately excluded, and the README should say so: torch, langchain/llama-index (would hide the
retrieval mechanics this project exists to measure), faiss, any vector DB server.

### Repo layout

```
aeration-qa/
├── config.yaml                 # single source of truth: model ids, chunk params, k, seeds
├── requirements.txt            # pymupdf, fastembed, numpy, rank-bm25, requests, pyyaml,
│                               #   pandas+matplotlib (eval scripts only), pytest
├── data/
│   ├── papers/                 # GITIGNORED — the PDFs
│   ├── MANIFEST.csv            # committed: paper_id, title, year, doi, url, license,
│   │                           #   open_access, filename, sha256, substitution reason
│   └── glossary.yaml           # committed: domain synonyms ("ammonia valley" ↔ "NH4-N
│                               #   breakpoint" ↔ "pH valley") for the query-expansion condition
├── cache/                      # GITIGNORED — extracted text, chunks.jsonl, embeddings.npy
├── src/aeration_qa/
│   ├── config.py               # loads config.yaml into a frozen dataclass + config hash
│   ├── extract.py              # PDF → per-page text (column sort, header/footer strip,
│   │                           #   de-hyphenation, references cutoff, section hints)
│   ├── chunking.py             # sentence-aware packing → Chunk objects
│   ├── embed.py                # fastembed wrapper; doc vs. query prefix handling
│   ├── index.py                # build/load/validate cache, staleness detection
│   ├── retrieve.py             # BM25 / dense / RRF hybrid behind one Retriever protocol
│   ├── generate.py             # prompt assembly, Ollama call, citation post-processing
│   └── pipeline.py             # ask(question, condition) → Answer; generation is a
│                               #   SWAPPABLE STRATEGY from day one (see fallback below)
├── scripts/
│   ├── 00_smoke_test.py        # GATE 0 — hardware feasibility, run before anything else
│   ├── 01_build_index.py       # PDFs → cache/, idempotent
│   └── 02_inspect_chunks.py    # dump random chunks for human QA
├── tests/                      # test_chunking.py, test_embed_prefix.py
├── ask.py                      # CLI: python ask.py "why does ORP show a nitrate knee?"
├── CLAUDE.md                   # written FIRST (see below)
└── README.md
```

### `CLAUDE.md` — written in Phase 0, before any code

So that a future Claude Code session opened in this folder understands the project without the user
re-explaining it. It must carry the things that are **not** derivable from reading the code:

- **What this is and why it exists** — a RAG QA assistant over aeration-control literature; it came out
  of the user's research into four families of aeration-control approaches; it is a genuine
  language-model project, unlike the paused `sbr-transition-transformer/` (a numeric time-series model
  that merely uses the transformer architecture) living in the user's `water-quality-forecasting` repo.
- **The hard constraints, stated as rules** — CPU only, 7.7 GB RAM (~3–4 GB usable), no paid API ever,
  Windows + Python 3.13. A future session must not "helpfully" propose fine-tuning, torch,
  `sentence-transformers`, a 7B model, or a hosted embedding service. Each ban needs its reason
  attached, or it will be re-litigated.
- **Why each dependency was chosen and what was deliberately excluded** — especially that
  langchain/llama-index are banned because they would hide the retrieval mechanics this project exists
  to measure, and that PyMuPDF's AGPL licence was a conscious accepted tradeoff.
- **The evaluation philosophy** — BM25 is a strong baseline, not a strawman; results are reported with
  paired bootstrap CIs; LLM-as-judge is refused because the only available judge is the model under
  test. This culture is inherited from the user's `water-quality-forecasting` repo.
- **The RAM choreography rule** — embedding and generation never run at the same time.
- **The BGE query-prefix trap** — queries need the prefix, passages don't; getting this wrong silently
  destroys dense retrieval.
- **Current status and what's next**, kept up to date as phases complete.

Eval scaffolding (`eval/`, `results/`, `metrics.py`, scripts 03–07) comes in a **second phase** — the
decision was to build a working pipeline first and size the evaluation after seeing it run. The
retrieval layer is still built with all three conditions from the start, since the `Retriever`
interface must support them and they cost almost nothing.

### Ingestion details that matter

- **References cutoff is not optional.** A references section is ~15% of a paper and pure author/journal
  noise that will dominate BM25 term statistics.
- **Two-column reconstruction:** cluster block `x0` values; if bimodal, sort by `(column, y0)`.
- **Scanned PDFs:** if a page yields <100 chars, abort with a clear message and substitute the paper.
  No OCR path — Tesseract on Windows is an installer + PATH + language data for one or two papers.
- **Chunking: 300 tokens, 60-token overlap, sentence-aligned.** BGE-small silently truncates at 512,
  so oversized chunks would lose their tail from the embedding while BM25 still sees all of it —
  quietly biasing the central comparison. 300 tokens also keeps prefill latency tolerable (four chunks
  ≈ 1,400 context tokens ≈ 15–30 s of prefill on this CPU, before the first output token).
- **Chunk records store `text` and `embed_text` separately:** dense embeds `title | section\n\ntext`,
  BM25 indexes bare `text` (prepending titles would inflate their term frequency across every chunk).
  Document this asymmetry.

### Retrieval

All conditions implement one protocol so nothing is special-cased:

```python
@dataclass(frozen=True)
class Hit: chunk_id: str; score: float; rank: int

class Retriever(Protocol):
    name: str
    def search(self, query: str, k: int) -> list[Hit]: ...
```

- **BM25**: lowercase, `re.findall(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")` so `NH4-N` and `DO` survive as
  tokens; ~40-word hardcoded stopword list; **no stemming** (a hidden free parameter); library-default
  `k1`/`b`, untuned.
- **Dense**: L2-normalized at build time, so search is `embeddings @ q` + `argpartition`.
  **BGE requires the query prefix** `"Represent this sentence for searching relevant passages: "` on
  queries but not passages — fastembed applies it in `query_embed()` but not `embed()`. Using `embed()`
  for both is the standard way people silently cripple BGE; `tests/test_embed_prefix.py` asserts the
  two paths differ for identical input.
- **Hybrid**: reciprocal rank fusion, `score = Σ 1/(60 + rank)`, over top-20 from each. RRF rather than
  score normalization because BM25 scores are unbounded and any normalization scheme becomes another
  untested parameter.

Expect BM25 to be **strong, not dumb**, on a tiny corpus dense in rare jargon (*ammonia valley*,
*nitrate knee*, *DO elbow*). "BM25 beat the embeddings" is a legitimate headline, not a failure.

### Generation

- System prompt: answer only from the excerpts, cite every claim as `[n]`, and reply exactly
  `NOT_IN_CORPUS` when the excerpts don't cover it (a literal sentinel makes abstention
  machine-checkable later).
- Context blocks numbered and headed `[n] {title} ({year}), p.{page}`; **highest-scoring chunk placed
  last**, right before the question — small models over-attend to the end of the context.
- Ollama `/api/chat`, `stream=False`, `temperature=0`, fixed `seed`, and **`num_ctx=4096` set
  explicitly on every request** (server defaults have changed across versions; silently getting 2048
  would truncate context and look like a retrieval failure).
- Record `prompt_eval_count/duration` and `eval_count/duration` per call — reporting prefill and decode
  throughput separately is a genuinely interesting result for CPU-only RAG.
- **Citations are not trusted to the model:** a Sources block is always appended from chunk metadata by
  code, so provenance survives even if the model cites nothing; inline `[n]` markers are regex-validated
  against the blocks actually sent, and out-of-range markers are counted as a hallucination signal.

### RAM choreography (the real architectural constraint)

Ollama is a separate resident process; fastembed holds an ONNX session. Both at once on a 3–4 GB budget
is tight, so: **embedding and generation never overlap** (script 01 embeds and exits; the interactive
path embeds one query vector then generates), `OLLAMA_MAX_LOADED_MODELS=1`, `OLLAMA_MODELS` pointed at
`D:\` , and long `keep_alive` during batch runs so the model loads once. Watch **commit charge**, not
just working set — once Windows pages the model file, throughput collapses and it looks like "the model
got slow" rather than "out of RAM."

## Verification

**Gate 0 — hardware feasibility, before writing anything else.** `scripts/00_smoke_test.py`: install
Ollama, pull the 3B, measure cold load time, prefill tok/s on a ~1,500-token prompt, decode tok/s,
end-to-end latency for one RAG-shaped call, peak RSS and system commit charge, at `num_ctx` 2048 and
4096. Commit the JSON — it justifies every downstream constraint.

| Result | Decision |
|---|---|
| < 45 s/question, no paging | proceed with 3B |
| 45–120 s | drop to 1.5B and re-measure |
| > 120 s, or paging | 1.5B; if that also fails → **extractive fallback** |

**Extractive fallback is not a consolation prize:** retrieve top-k, then select the 2–3 most
query-relevant *sentences* and present them verbatim with citations. Fast, fully grounded, cannot
hallucinate — and the entire retrieval layer, which is where this project's value lives, is unaffected.
This is why `pipeline.py` treats generation as a swappable strategy from day one.

Remaining gates, in order:

1. **Extraction** — run `01_build_index.py` on 2 papers and *read the extracted text by eye*: column
   order, references truncated, no running heads, no mojibake. Fix before scaling to 10 papers.
2. **Chunking** — `02_inspect_chunks.py` on 20 random chunks: no mid-sentence starts, none over the
   512-token cap, overlap present, page numbers match the printed PDF. Plus `pytest tests/`.
3. **Embeddings** — embed 5 hand-written questions whose answers you know and print top-3. If
   "What is the ammonia valley?" doesn't surface an ammonia-valley passage, the query prefix is missing.
4. **End-to-end** — `python ask.py "..."` on ~10 questions, spot-checked by hand, comparing BM25 vs
   dense vs hybrid output for the same question.

## Sequencing

- **Phase 0 (half day):** `git init`, **write `CLAUDE.md` first**, venv,
  `pip install -r requirements.txt` (immediately confirms the Python 3.13 wheel story), install Ollama,
  pull model, **Gate 0**.
- **Phase 1 (~1 day, mostly non-coding):** collect 10–12 PDFs, fill `MANIFEST.csv` (prefer open access;
  the Taylor & Francis 2025 article is likely paywalled — record substitutions and why), write
  `glossary.yaml` while the literature is fresh.
- **Phase 2 (1–2 days):** `config.py`, `extract.py`, `chunking.py`, inspection script, tests. Gates 1–2.
- **Phase 3 (1 day):** `embed.py`, `index.py`, `retrieve.py`. Gate 3 — you now have a working search
  engine.
- **Phase 4 (1–2 days):** `generate.py`, `pipeline.py`, `ask.py`. Gate 4.
- **Phase 5 (deferred; scope decided after seeing Phase 4):** the evaluation harness — gold
  question set with stable `(paper_id, page, snippet)` anchors so re-chunking doesn't invalidate it, a
  pre-registered `PROTOCOL.md` committed *before* results exist, retrieval metrics (nDCG@10 primary),
  an **oracle condition** feeding gold passages straight to the generator to separate retrieval failure
  from generation failure, and paired bootstrap CIs. Explicitly **no LLM-as-judge** — the only available
  judge is the same weak model under evaluation.

- **Phase 6 (future extension, only after Phases 0–5 work):** fold the paused bending-point detector
  back in as a *second information source* alongside the papers. Given a sensor series for a cycle, a
  detector computes where the DO elbow / ammonia valley / nitrate knee / apex fall, and that numeric
  result is injected into the prompt as one more numbered block beside the retrieved passages, so the
  model can answer questions like *"DO on tank 5.7 looks like this — is it time to stop aeration?"*
  with both a computed answer and literature grounding.

  **Design rule for this phase: the model must not decide when to call the detector.** Small local
  models are unreliable at function calling — they invoke the wrong tool, pass wrong arguments, or skip
  the call entirely. Instead, application code always runs the detector when sensor data is present and
  passes the result in as context. The LLM only interprets and explains; it decides nothing. This keeps
  arithmetic in code (where it is exact) and language in the model (where it belongs).

  Blocked on a real sensor data source. The plant historian endpoint investigated on 2026-09-20
  (`ai.sfcwater.com/api/backend/api/history`) returns an empty array for every tag and time range, while
  the live PLC endpoint on the same backend returns real values — so the fault is isolated to the
  archive path and is the platform operator's to fix. Until then this phase can only run on synthetic
  cycles from `SBRCycleSimulator`.

## Constraint to respect throughout

**No git operations of any kind on `dnulab/onboarding`** (the shared lab repo) — the user has
explicitly forbidden it. This project is a separate repo; the paused SBR subproject lives in the user's
own private `water-quality-forecasting`.
